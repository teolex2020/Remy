"""Small explainable multilingual reranker that runs entirely on-device.

This is intentionally not a bundled neural model. It has no API, account,
model download, GPU, or language-specific tokenizer requirement. Unicode-aware
token similarity, query-local inverse-document-frequency weights, and field
coverage provide a stronger second-stage ranking than raw engine position while
remaining cheap enough for every chat search.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any


_TOKEN_RE = re.compile(r"[^\W_]+(?:['’ʼ][^\W_]+)?", re.UNICODE)
_URL_SEPARATOR_RE = re.compile(r"[/_.?=&:#-]+")
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'"})

# Function words only. Product and technical terms must never be discarded.
_STOP_WORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "with",
    # Ukrainian
    "або", "але", "без", "був", "була", "були", "від", "для", "до",
    "за", "з", "із", "і", "й", "на", "не", "про", "та", "у", "це",
    "чи", "що", "як", "яка", "яке", "який",
    # Russian
    "без", "был", "была", "были", "в", "для", "до", "и", "из", "как",
    "на", "не", "о", "от", "по", "с", "что", "это",
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return " ".join(normalized.translate(_APOSTROPHES).split())


def multilingual_tokens(value: str, *, url: bool = False) -> list[str]:
    normalized = normalize_text(value)
    if url:
        normalized = _URL_SEPARATOR_RE.sub(" ", normalized)
    return [
        token.strip("'")
        for token in _TOKEN_RE.findall(normalized)
        if len(token.strip("'")) >= 2 and token.strip("'") not in _STOP_WORDS
    ]


def _char_ngrams(token: str, size: int = 3) -> set[str]:
    if len(token) <= size:
        return {token}
    return {token[index : index + size] for index in range(len(token) - size + 1)}


def token_similarity(query_token: str, document_token: str) -> float:
    """Return conservative language-agnostic similarity in ``[0, 1]``."""
    if query_token == document_token:
        return 1.0
    shorter, longer = sorted((query_token, document_token), key=len)
    if len(shorter) >= 4 and longer.startswith(shorter):
        return 0.94
    if min(len(query_token), len(document_token)) < 4:
        return 0.0

    common_prefix = 0
    for left, right in zip(query_token, document_token):
        if left != right:
            break
        common_prefix += 1
    has_cyrillic = any("\u0400" <= character <= "\u04ff" for character in shorter)
    required_prefix = 3 if has_cyrillic else 4
    prefix_ratio = common_prefix / len(shorter)
    if common_prefix >= required_prefix and prefix_ratio >= 0.70:
        # Lightweight morphology support: Ukrainian/Russian case endings and
        # common inflections usually preserve a long lexical stem.
        return round(min(0.94, 0.78 + prefix_ratio * 0.20), 4)

    sequence_ratio = SequenceMatcher(None, query_token, document_token).ratio()
    query_grams = _char_ngrams(query_token)
    document_grams = _char_ngrams(document_token)
    dice = (
        2 * len(query_grams & document_grams) / (len(query_grams) + len(document_grams))
        if query_grams and document_grams
        else 0.0
    )
    similarity = max(sequence_ratio, dice)
    # Below this boundary, common character fragments create more noise than
    # useful cross-language morphology/typo recovery.
    return round(similarity, 4) if similarity >= 0.72 else 0.0


def _best_match(term: str, tokens: Sequence[str]) -> tuple[float, str]:
    best_score = 0.0
    best_token = ""
    for token in tokens:
        score = token_similarity(term, token)
        if score > best_score:
            best_score = score
            best_token = token
            if score == 1.0:
                break
    return best_score, best_token


def _ordered_coverage(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
    if len(query_tokens) < 2 or not document_tokens:
        return 0.0
    cursor = 0
    matched_positions: list[int] = []
    for term in query_tokens:
        found = None
        for position in range(cursor, len(document_tokens)):
            if token_similarity(term, document_tokens[position]) >= 0.86:
                found = position
                break
        if found is not None:
            matched_positions.append(found)
            cursor = found + 1
    if len(matched_positions) < 2:
        return 0.0
    coverage = len(matched_positions) / len(query_tokens)
    span = matched_positions[-1] - matched_positions[0] + 1
    compactness = min(1.0, len(matched_positions) / max(1, span))
    return round(coverage * (0.65 + compactness * 0.35), 4)


class LocalMultilingualReranker:
    """Score a candidate set using query-local, explainable lexical evidence."""

    name = "local:multilingual-idf-fuzzy-v1"

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        query_tokens = list(dict.fromkeys(multilingual_tokens(query)))[:24]
        if not query_tokens:
            return [self._empty_score("query_has_no_distinctive_terms") for _ in candidates]

        documents: list[dict[str, list[str]]] = []
        for candidate in candidates:
            fields = {
                "title": multilingual_tokens(str(candidate.get("title") or "")),
                "snippet": multilingual_tokens(str(candidate.get("snippet") or "")),
                "url": multilingual_tokens(str(candidate.get("uri") or ""), url=True),
            }
            fields["all"] = list(dict.fromkeys(fields["title"] + fields["snippet"] + fields["url"]))
            documents.append(fields)

        document_count = max(1, len(documents))
        document_frequency: Counter[str] = Counter()
        for term in query_tokens:
            document_frequency[term] = sum(
                1
                for fields in documents
                if _best_match(term, fields["all"])[0] >= 0.86
            )
        idf = {
            term: math.log(1 + (document_count - document_frequency[term] + 0.5)
                           / (document_frequency[term] + 0.5))
            for term in query_tokens
        }
        total_idf = sum(idf.values()) or 1.0

        scores: list[dict[str, Any]] = []
        for fields in documents:
            matched_exact: list[str] = []
            matched_fuzzy: dict[str, str] = {}
            missing: list[str] = []
            weighted_coverage = 0.0
            weighted_title = 0.0

            for term in query_tokens:
                title_score, title_token = _best_match(term, fields["title"])
                snippet_score, snippet_token = _best_match(term, fields["snippet"])
                url_score, url_token = _best_match(term, fields["url"])
                field_score, field_token = max(
                    (title_score, title_token),
                    (snippet_score * 0.82, snippet_token),
                    (url_score * 0.62, url_token),
                    key=lambda item: item[0],
                )
                weighted_coverage += idf[term] * field_score
                weighted_title += idf[term] * title_score
                raw_best = max(title_score, snippet_score, url_score)
                best_token = (
                    title_token if title_score == raw_best
                    else snippet_token if snippet_score == raw_best
                    else url_token
                )
                if raw_best == 1.0:
                    matched_exact.append(term)
                elif raw_best >= 0.72:
                    matched_fuzzy[term] = best_token
                else:
                    missing.append(term)

            coverage = weighted_coverage / total_idf
            title_coverage = weighted_title / total_idf
            ordered = max(
                _ordered_coverage(query_tokens, fields["title"]),
                _ordered_coverage(query_tokens, fields["snippet"]) * 0.8,
            )
            score = min(1.0, coverage * 0.72 + title_coverage * 0.20 + ordered * 0.08)
            scores.append(
                {
                    "name": self.name,
                    "score": round(score, 4),
                    "coverage": round(coverage, 4),
                    "title_coverage": round(title_coverage, 4),
                    "ordered_coverage": round(ordered, 4),
                    "matched_exact": matched_exact,
                    "matched_fuzzy": matched_fuzzy,
                    "missing_terms": missing,
                }
            )
        return scores

    def _empty_score(self, reason: str) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": 0.5,
            "coverage": 1.0,
            "title_coverage": 1.0,
            "ordered_coverage": 0.0,
            "matched_exact": [],
            "matched_fuzzy": {},
            "missing_terms": [],
            "reason": reason,
        }


_default_reranker: LocalMultilingualReranker | None = None


def get_local_multilingual_reranker() -> LocalMultilingualReranker:
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = LocalMultilingualReranker()
    return _default_reranker
