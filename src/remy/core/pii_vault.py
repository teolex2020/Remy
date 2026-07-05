"""
PII Shield — Privacy layer that tokenizes personal data before LLM API calls.

Replaces sensitive values (names, phones, tracked metrics, etc.) with opaque
tokens like [PII:name_1] before sending to Gemini, and restores real values
in the response before showing to the user.

Session-scoped: each session gets its own vault, destroyed on close.
"""

import logging
import re
import threading

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger("PII")

# ══════════════════════════════════════════════════════════════════════
# Reuse regex patterns from provenance.py
# ══════════════════════════════════════════════════════════════════════
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_WALLET_RE = re.compile(r"0x[a-fA-F0-9]{40}")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-\(\)]{7,}\d)")
_PASSWORD_RE = re.compile(r"(?:password|пароль|pass|pwd)\s*[:=]\s*\S+", re.IGNORECASE)
_API_KEY_RE = re.compile(r"(?:api.?key|token|secret)\s*[:=]\s*\S+", re.IGNORECASE)

# Health metric patterns
_BP_RE = re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b")  # 120/80, 135 / 90
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_CARD_RE = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")

# Category prefixes for token naming
_CAT_NAME = "name"
_CAT_PHONE = "phone"
_CAT_EMAIL = "email"
_CAT_ADDR = "addr"
_CAT_BP = "bp"
_CAT_HEALTH = "health"
_CAT_WALLET = "wallet"
_CAT_FIN = "fin"
_CAT_SECRET = "secret"

# LLM hint prepended to system instruction when PII shield is active
PII_HINT = (
    "[PII Shield Active] Some values are replaced with [PII:*] tokens for privacy.\n"
    "Token meanings: name=person name, phone=phone number, email=email, "
    "addr=address, bp=blood pressure, health=health value, "
    "wallet=crypto wallet, fin=financial data, secret=password/key.\n"
    "Treat tokens as real values. Use them naturally in your response.\n\n"
)


def _clean_profile_value(value) -> str:
    """Normalize optional profile metadata values before PII indexing."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


# ══════════════════════════════════════════════════════════════════════
# Vault — session-scoped bidirectional mapping
# ══════════════════════════════════════════════════════════════════════


class PIIVault:
    """Session-scoped PII token ↔ real value mapping."""

    def __init__(self):
        self._to_token: dict[str, str] = {}  # real_value → token
        self._to_value: dict[str, str] = {}  # token → real_value
        self._counters: dict[str, int] = {}  # category → next_id
        self._lock = threading.Lock()
        # Known profile values loaded once per session
        self._profile_loaded = False
        self._profile_values: dict[str, str] = {}  # value → category

    def _next_token(self, category: str) -> str:
        n = self._counters.get(category, 0) + 1
        self._counters[category] = n
        return f"[PII:{category}_{n}]"

    def tokenize(self, value: str, category: str) -> str:
        """Replace a real value with a PII token. Reuses existing token if seen before."""
        if not value or len(value.strip()) < 2:
            return value
        with self._lock:
            key = value.strip()
            if key in self._to_token:
                return self._to_token[key]
            token = self._next_token(category)
            self._to_token[key] = token
            self._to_value[token] = key
            return token

    def restore_text(self, text: str) -> str:
        """Replace all PII tokens in text with real values.

        Some models occasionally mutate the token punctuation in streamed text,
        e.g. [PII:name_2] -> [PI!name_2]. Treat those as display aliases only;
        the vault's canonical mapping remains [PII:*].
        """
        if not text or "[PI" not in text:
            return text
        with self._lock:
            for token, value in self._to_value.items():
                text = text.replace(token, value)
                if token.startswith("[PII:") and token.endswith("]"):
                    body = token[len("[PII:"):-1]
                    for alias in (
                        f"[PI!{body}]",
                        f"[PII!{body}]",
                        f"[PII {body}]",
                        f"[PII-{body}]",
                    ):
                        text = text.replace(alias, value)
        return text

    def load_profile(self):
        """Load known PII values from user profile (called once per session)."""
        if self._profile_loaded:
            return
        self._profile_loaded = True
        try:
            from remy.core.agent_tools import brain, brain_lock

            from remy.core.brain_tools import get_user_profile_record
            _profile = get_user_profile_record(brain, brain_lock)
            if not _profile:
                return
            meta = _profile.metadata or {}

            # Map profile fields to PII categories
            _field_to_cat = {
                "name": _CAT_NAME,
                "phone": _CAT_PHONE,
                "email": _CAT_EMAIL,
                "location": _CAT_ADDR,
                "family": _CAT_NAME,
                "occupation": _CAT_NAME,
            }
            for field, cat in _field_to_cat.items():
                val = _clean_profile_value(meta.get(field))
                if val and len(val) >= 2:
                    self._profile_values[val] = cat
                    # Also add individual family members
                    if field == "family" and "," in val:
                        for part in val.split(","):
                            part = part.strip()
                            if len(part) >= 2:
                                self._profile_values[part] = _CAT_NAME

            # Load person records for family names
            try:
                with brain_lock:
                    persons = brain.search(query="", tags=["person"], limit=20)
                for p in persons:
                    pmeta = p.metadata or {}
                    pname = _clean_profile_value(pmeta.get("name"))
                    if pname and len(pname) >= 2:
                        self._profile_values[pname] = _CAT_NAME
            except Exception:
                pass

            logger.debug("PII vault loaded %d profile values", len(self._profile_values))
        except Exception as e:
            logger.warning("Failed to load profile for PII vault: %s", e)


# ══════════════════════════════════════════════════════════════════════
# Session registry — one vault per session
# ══════════════════════════════════════════════════════════════════════

_vaults: dict[str, PIIVault] = {}
_vaults_lock = threading.Lock()


def get_vault(session_id: str) -> PIIVault:
    """Get or create a PII vault for the given session."""
    with _vaults_lock:
        if session_id not in _vaults:
            _vaults[session_id] = PIIVault()
        return _vaults[session_id]


def destroy_vault(session_id: str):
    """Remove and garbage-collect vault for a closed session."""
    with _vaults_lock:
        _vaults.pop(session_id, None)
    logger.debug("PII vault destroyed for session %s", session_id[:8])


# ══════════════════════════════════════════════════════════════════════
# Shield — detect and replace PII in text
# ══════════════════════════════════════════════════════════════════════


def shield(text: str, vault: PIIVault) -> str:
    """Replace PII in text with tokens using the vault.

    Detection priority:
    1. Known profile values (exact match, longest first)
    2. Regex patterns (email, wallet, phone, password, API key)
    3. Health metric patterns (blood pressure, IBAN, card numbers)
    """
    if not text:
        return text

    # Ensure profile is loaded
    vault.load_profile()

    # 1. Known profile values — longest first to avoid partial matches
    for value in sorted(vault._profile_values.keys(), key=len, reverse=True):
        if value in text:
            cat = vault._profile_values[value]
            token = vault.tokenize(value, cat)
            text = text.replace(value, token)

    # 2. Regex-based detection
    # Emails
    for m in _EMAIL_RE.finditer(text):
        text = text.replace(m.group(), vault.tokenize(m.group(), _CAT_EMAIL))

    # Crypto wallets
    for m in _WALLET_RE.finditer(text):
        text = text.replace(m.group(), vault.tokenize(m.group(), _CAT_WALLET))

    # Phone numbers
    for m in _PHONE_RE.finditer(text):
        text = text.replace(m.group(), vault.tokenize(m.group(), _CAT_PHONE))

    # Passwords / API keys
    for m in _PASSWORD_RE.finditer(text):
        text = text.replace(m.group(), vault.tokenize(m.group(), _CAT_SECRET))
    for m in _API_KEY_RE.finditer(text):
        text = text.replace(m.group(), vault.tokenize(m.group(), _CAT_SECRET))

    # 3. Health metrics
    # Blood pressure: 120/80
    for m in _BP_RE.finditer(text):
        text = text.replace(m.group(), vault.tokenize(m.group(), _CAT_BP))

    # IBAN
    for m in _IBAN_RE.finditer(text):
        text = text.replace(m.group(), vault.tokenize(m.group(), _CAT_FIN))

    # Card numbers
    for m in _CARD_RE.finditer(text):
        text = text.replace(m.group(), vault.tokenize(m.group(), _CAT_FIN))

    return text


def restore(text: str, vault: PIIVault) -> str:
    """Replace PII tokens in text with real values."""
    return vault.restore_text(text)


# ══════════════════════════════════════════════════════════════════════
# Message-level shield — process full LangChain message lists
# ══════════════════════════════════════════════════════════════════════


def shield_messages(messages: list[BaseMessage], vault: PIIVault) -> list[BaseMessage]:
    """Shield PII in a list of LangChain messages before sending to LLM.

    Returns a new list with tokenized content. Original messages are not mutated.
    Adds PII hint to system instruction.
    """
    result = []
    for i, msg in enumerate(messages):
        content = msg.content
        if isinstance(content, str):
            shielded = shield(content, vault)
            # Prepend PII hint to system instruction
            if i == 0 and isinstance(msg, SystemMessage):
                shielded = PII_HINT + shielded
        elif isinstance(content, list):
            # Multimodal messages (list of dicts with text/image parts)
            shielded = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    shielded.append({**part, "text": shield(part["text"], vault)})
                elif isinstance(part, str):
                    shielded.append(shield(part, vault))
                else:
                    shielded.append(part)
        else:
            shielded = content

        # Create new message of same type with shielded content
        if isinstance(msg, SystemMessage):
            result.append(SystemMessage(content=shielded))
        elif isinstance(msg, HumanMessage):
            result.append(HumanMessage(content=shielded))
        elif isinstance(msg, AIMessage):
            result.append(
                AIMessage(
                    content=shielded,
                    tool_calls=msg.tool_calls if hasattr(msg, "tool_calls") else [],
                    id=msg.id,
                )
            )
        elif isinstance(msg, ToolMessage):
            result.append(
                ToolMessage(
                    content=shielded,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", ""),
                )
            )
        else:
            result.append(msg)

    return result


# ══════════════════════════════════════════════════════════════════════
# Streaming restorer — handles partial PII tokens across chunks
# ══════════════════════════════════════════════════════════════════════


class StreamingRestorer:
    """Buffers streaming chunks to correctly restore split PII tokens.

    Usage:
        restorer = StreamingRestorer(vault)
        for chunk in llm_stream:
            ready = restorer.feed(chunk)
            if ready:
                yield ready
        # At end of stream:
        remaining = restorer.flush()
        if remaining:
            yield remaining
    """

    def __init__(self, vault: PIIVault):
        self._vault = vault
        self._buffer = ""

    _PREFIX = "[PII:"
    _PARTIAL_PREFIXES = ("[PII:", "[PII", "[PI", "[P", "[")

    def feed(self, chunk: str) -> str:
        """Add a chunk, return text safe to yield (fully restored)."""
        self._buffer += chunk

        # Check for complete PII tokens first — restore them in-place
        self._buffer = self._vault.restore_text(self._buffer)

        # Check if buffer ends with a partial PII token prefix
        # e.g. buffer ends with "[P" or "[PII:" — hold it back
        for prefix in self._PARTIAL_PREFIXES:
            if self._buffer.endswith(prefix):
                ready = self._buffer[: -len(prefix)]
                self._buffer = prefix
                return ready

        # Also check for incomplete [PII:xxx (no closing bracket yet)
        last_bracket = self._buffer.rfind("[PII:")
        if last_bracket >= 0:
            closing = self._buffer.find("]", last_bracket)
            if closing < 0:
                ready = self._buffer[:last_bracket]
                self._buffer = self._buffer[last_bracket:]
                return ready

        # No partial tokens — yield everything
        ready = self._buffer
        self._buffer = ""
        return ready

    def flush(self) -> str:
        """Flush remaining buffer (end of stream). Returns restored text."""
        remaining = self._buffer
        self._buffer = ""
        return self._vault.restore_text(remaining)
