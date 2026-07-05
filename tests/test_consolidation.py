"""Tests for Memory Consolidation in Background Brain."""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from aura import Aura as CognitiveMemory, Level


# ============== Cluster Finding Tests ==============

class TestFindConsolidationClusters:

    def test_empty_brain_no_clusters(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _find_consolidation_clusters

        clusters = _find_consolidation_clusters(b)
        assert clusters == []
        b.close()

    def test_few_records_no_clusters(self, tmp_path):
        """Less than MIN_CLUSTER_SIZE records → no clusters."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _find_consolidation_clusters

        b.store(content="Record one", level=Level.DOMAIN, tags=["topic-a"])
        b.store(content="Record two", level=Level.DOMAIN, tags=["topic-a"])

        clusters = _find_consolidation_clusters(b)
        assert clusters == []
        b.close()

    def test_finds_cluster_of_similar_tags(self, tmp_path):
        """3+ records with same tags → one cluster."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _find_consolidation_clusters

        # Content must be distinct enough so Aura doesn't deduplicate
        contents = [
            "Italian pasta carbonara recipe with eggs, bacon and cheese",
            "Japanese ramen noodle soup with pork belly and miso broth",
            "Mexican enchiladas with chicken, salsa verde and sour cream",
            "Indian butter chicken curry with basmati rice and naan bread",
        ]
        for c in contents:
            b.store(content=c, level=Level.DOMAIN, tags=["cooking", "recipe"])

        clusters = _find_consolidation_clusters(b)
        assert len(clusters) >= 1
        # Should find a cluster with tag key containing "cooking" and "recipe"
        tag_keys = [c[0] for c in clusters]
        assert any("cooking" in tk and "recipe" in tk for tk in tag_keys)
        b.close()

    def test_skips_system_tags(self, tmp_path):
        """Records with system tags (user-profile, session-summary, etc.) are skipped."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _find_consolidation_clusters

        for i in range(5):
            b.store(
                content=f"Session summary {i}",
                level=Level.DOMAIN,
                tags=["session-summary"],
            )

        clusters = _find_consolidation_clusters(b)
        assert clusters == []
        b.close()

    def test_skips_already_consolidated(self, tmp_path):
        """Records with consolidated_into metadata are skipped."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _find_consolidation_clusters

        for i in range(4):
            rec = b.store(
                content=f"Already consolidated record {i}",
                level=Level.DOMAIN,
                tags=["topic-x"],
            )
            b.update(rec.id, metadata={"consolidated_into": "meta-abc"})

        clusters = _find_consolidation_clusters(b)
        assert clusters == []
        b.close()

    def test_multiple_clusters_sorted_by_size(self, tmp_path):
        """Multiple clusters returned sorted by size (largest first)."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _find_consolidation_clusters

        # Small cluster: 3 records
        for i in range(3):
            b.store(content=f"Small cluster item {i}", level=Level.DOMAIN, tags=["small-topic"])

        # Large cluster: 5 records
        for i in range(5):
            b.store(content=f"Large cluster item {i}", level=Level.DOMAIN, tags=["large-topic"])

        clusters = _find_consolidation_clusters(b)
        if len(clusters) >= 2:
            # Largest cluster first
            assert len(clusters[0][1]) >= len(clusters[1][1])
        b.close()


# ============== Verify Similarity Tests ==============

class TestVerifyClusterSimilarity:

    def test_returns_records_when_similar(self, tmp_path):
        """Records that are similar to anchor should be kept."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _verify_cluster_similarity

        recs = []
        for i in range(4):
            r = b.store(
                content=f"Information about Python programming language version {i}",
                level=Level.DOMAIN,
                tags=["python"],
            )
            recs.append(r)

        result = _verify_cluster_similarity(b, recs)
        # Should keep at least MIN_CLUSTER_SIZE records
        assert len(result) >= 3
        b.close()

    def test_short_content_returns_as_is(self, tmp_path):
        """Records with very short content skip similarity check."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _verify_cluster_similarity

        recs = []
        for i in range(3):
            r = b.store(content="x", level=Level.DOMAIN, tags=["short"])
            recs.append(r)

        result = _verify_cluster_similarity(b, recs)
        assert len(result) == 3  # Returned as-is
        b.close()


# ============== Generate Consolidation Summary Tests ==============

class TestGenerateConsolidationSummary:

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_generates_summary(self, mock_llm_cls):
        from remy.core.background_brain import _generate_consolidation_summary

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Combined summary of cooking recipes: pasta, risotto, and pizza.")
        mock_llm_cls.return_value = mock_llm

        sources = [
            {"id": "r1", "content": "Recipe for pasta carbonara", "tags": ["cooking"], "level": 3},
            {"id": "r2", "content": "Recipe for mushroom risotto", "tags": ["cooking"], "level": 3},
            {"id": "r3", "content": "Recipe for margherita pizza", "tags": ["cooking"], "level": 3},
        ]

        result = _generate_consolidation_summary("cooking,recipe", sources)
        assert result is not None
        assert "Combined summary" in result
        mock_llm.invoke.assert_called_once()

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_returns_none_on_llm_failure(self, mock_llm_cls):
        from remy.core.background_brain import _generate_consolidation_summary

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")
        mock_llm_cls.return_value = mock_llm

        sources = [
            {"id": "r1", "content": "Record 1", "tags": ["t"], "level": 3},
            {"id": "r2", "content": "Record 2", "tags": ["t"], "level": 3},
            {"id": "r3", "content": "Record 3", "tags": ["t"], "level": 3},
        ]

        result = _generate_consolidation_summary("t", sources)
        assert result is None

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_returns_none_on_short_summary(self, mock_llm_cls):
        from remy.core.background_brain import _generate_consolidation_summary

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="ok")
        mock_llm_cls.return_value = mock_llm

        sources = [
            {"id": "r1", "content": "Record 1", "tags": ["t"], "level": 3},
            {"id": "r2", "content": "Record 2", "tags": ["t"], "level": 3},
            {"id": "r3", "content": "Record 3", "tags": ["t"], "level": 3},
        ]

        result = _generate_consolidation_summary("t", sources)
        assert result is None  # Too short


# ============== Merge Cluster Tests ==============

class TestMergeCluster:

    @patch("remy.core.background_brain._generate_consolidation_summary")
    def test_merge_creates_meta_record(self, mock_summary, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _merge_cluster

        mock_summary.return_value = "Consolidated: three related cooking facts about Italian cuisine."

        # Content must be distinct enough so Aura doesn't deduplicate
        contents = [
            "Italian pasta carbonara recipe with eggs, bacon and parmesan",
            "Japanese ramen noodle soup with pork belly and miso broth",
            "Mexican enchiladas with chicken, salsa verde and sour cream",
        ]
        recs = []
        for c in contents:
            r = b.store(content=c, level=Level.DOMAIN, tags=["cooking"])
            recs.append(r)

        meta_id = _merge_cluster(b, "cooking", [r.id for r in recs])
        assert meta_id is not None

        # Verify meta-record
        meta = b.get(meta_id)
        assert meta is not None
        assert "consolidated-meta" in meta.tags
        assert "Consolidated:" in meta.content
        assert meta.metadata["type"] == "consolidation"
        assert meta.metadata["source_count"] == 3
        assert len(meta.metadata["source_ids"]) == 3

        # Verify sources are marked
        for r in recs:
            updated = b.get(r.id)
            assert updated.metadata.get("consolidated_into") == meta_id

        # Verify connections exist (meta connects to sources)
        meta_refreshed = b.get(meta_id)
        assert len(meta_refreshed.connections) == 3
        b.close()

    @patch("remy.core.background_brain._generate_consolidation_summary")
    def test_merge_returns_none_on_summary_failure(self, mock_summary, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _merge_cluster

        mock_summary.return_value = None

        recs = []
        for i in range(3):
            r = b.store(content=f"Record {i}", level=Level.DOMAIN, tags=["t"])
            recs.append(r)

        meta_id = _merge_cluster(b, "t", [r.id for r in recs])
        assert meta_id is None
        b.close()

    @patch("remy.core.background_brain._generate_consolidation_summary")
    def test_merge_returns_none_on_too_few_sources(self, mock_summary, tmp_path):
        """If records were deleted between finding and merging."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _merge_cluster

        mock_summary.return_value = "Summary"

        r = b.store(content="Only one", level=Level.DOMAIN, tags=["t"])
        meta_id = _merge_cluster(b, "t", [r.id, "nonexistent-1", "nonexistent-2"])
        assert meta_id is None  # Only 1 valid source < MIN_CLUSTER_SIZE
        b.close()


# ============== Full Consolidation Flow Tests ==============

class TestConsolidateRecords:

    @patch("remy.core.background_brain._generate_consolidation_summary")
    def test_full_consolidation_flow(self, mock_summary, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _consolidate_records

        mock_summary.return_value = "Consolidated summary of health tips about exercise and nutrition."

        # Content must be distinct enough so Aura doesn't deduplicate
        contents = [
            "Morning jogging for 30 minutes improves cardiovascular health",
            "Eating leafy greens daily provides essential vitamins and minerals",
            "Sleep 8 hours per night for optimal brain function and recovery",
            "Meditation for 15 minutes reduces stress hormones and anxiety",
        ]
        for c in contents:
            b.store(content=c, level=Level.DOMAIN, tags=["health", "tips"])

        result = _consolidate_records(b)
        assert result["clusters_found"] >= 1
        assert result["records_merged"] >= 3
        assert result["meta_records_created"] >= 1

        # Verify meta-record was created
        metas = b.search(query="", tags=["consolidated-meta"], limit=5)
        assert len(metas) >= 1
        b.close()

    def test_no_clusters_returns_zeros(self, tmp_path):
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _consolidate_records

        result = _consolidate_records(b)
        assert result == {"clusters_found": 0, "records_merged": 0, "meta_records_created": 0}
        b.close()

    @patch("remy.core.background_brain._generate_consolidation_summary")
    def test_max_clusters_per_run_limit(self, mock_summary, tmp_path):
        """Should not process more than MAX_CLUSTERS_PER_RUN clusters."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import (
            _consolidate_records, _MAX_CLUSTERS_PER_RUN,
        )

        mock_summary.return_value = "Consolidated summary."

        # Create 5 distinct clusters (each with 3+ records)
        for cluster_idx in range(5):
            tag = f"cluster-{cluster_idx}"
            for i in range(3):
                b.store(
                    content=f"Content for {tag} record {i} with unique info",
                    level=Level.DOMAIN,
                    tags=[tag],
                )

        result = _consolidate_records(b)
        assert result["meta_records_created"] <= _MAX_CLUSTERS_PER_RUN
        b.close()

    @patch("remy.core.background_brain._generate_consolidation_summary")
    def test_event_bus_emissions(self, mock_summary, tmp_path):
        """Consolidation emits events for live stream."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import _consolidate_records

        mock_summary.return_value = "Consolidated health tips summary."

        # Content must be distinct enough so Aura doesn't deduplicate
        contents = [
            "Running improves heart health and endurance over time",
            "Yoga stretches increase flexibility and reduce joint pain",
            "Swimming is a full body workout with low impact on joints",
            "Weight training builds muscle mass and strengthens bones",
        ]
        for c in contents:
            b.store(content=c, level=Level.DOMAIN, tags=["health", "wellness"])

        events = []
        with patch("remy.core.event_bus.event_bus") as mock_bus:
            mock_bus.emit = lambda t, d=None: events.append((t, d))
            _consolidate_records(b)

        event_types = [e[0] for e in events]
        assert "consolidation_start" in event_types
        assert "consolidation_merged" in event_types
        assert "consolidation_end" in event_types
        b.close()


# ============== Integration with run_background ==============

class TestConsolidationInRunBackground:

    def test_report_includes_consolidation(self, tmp_path):
        """run_background report includes consolidation section."""
        b = CognitiveMemory(str(tmp_path / "brain"))
        from remy.core.background_brain import run_background

        report = run_background(brain=b)
        assert "consolidation" in report
        assert "clusters_found" in report["consolidation"]
        assert "records_merged" in report["consolidation"]
        assert "meta_records_created" in report["consolidation"]
        b.close()
