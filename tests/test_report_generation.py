"""Tests for PDF report generation tool."""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============== TOOL DECLARATION ==============


def test_generate_report_tool_exists():
    """generate_report should be declared in BRAIN_TOOLS."""
    from remy.core.brain_tools import BRAIN_TOOLS

    names = [t.name for t in BRAIN_TOOLS]
    assert "generate_report" in names


def test_generate_report_tool_has_required_params():
    """generate_report should require title and allow content-only fallback."""
    from remy.core.brain_tools import BRAIN_TOOLS

    decl = next(t for t in BRAIN_TOOLS if t.name == "generate_report")
    assert "title" in decl.parameters.properties
    assert "content" in decl.parameters.properties
    assert "sections" in decl.parameters.properties
    assert "title" in decl.parameters.required
    assert "sections" not in decl.parameters.required


# ============== REPORT BUILDER ==============


class TestReportBuilder:

    def test_basic_report(self):
        """ReportBuilder creates a valid PDF file."""
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportBuilder(
                title="Test Report",
                subtitle="Unit Test",
                output_dir=tmpdir,
            )
            report.add_section("Introduction", "This is a test section.")
            filepath = report.save()

            assert Path(filepath).exists()
            assert filepath.endswith(".pdf")
            # Verify it's a real PDF (starts with %PDF)
            with open(filepath, "rb") as f:
                header = f.read(4)
            assert header == b"%PDF"

    def test_all_section_types(self):
        """ReportBuilder supports all section types without errors."""
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportBuilder(title="Full Test", output_dir=tmpdir)
            report.add_section("Section", "Body text")
            report.add_subsection("Subsection", "Sub body")
            report.add_text("Plain text paragraph")
            report.add_quote("A notable quote")
            report.add_key_findings(["Finding one", "Finding two"])
            report.add_table(
                headers=["Col A", "Col B"],
                rows=[["r1c1", "r1c2"], ["r2c1", "r2c2"]],
                title="Data Table",
            )
            report.add_memory_records([
                {
                    "id": "abc12345",
                    "content": "Test memory record",
                    "trust_score": 0.8,
                    "source": "agent-interactive",
                    "tags": ["test"],
                },
                {
                    "id": "def67890",
                    "content": "Low trust record",
                    "trust_score": 0.2,
                    "source": "agent-autonomous",
                    "tags": ["unverified"],
                },
            ])
            report.add_audit_summary([
                {
                    "timestamp": "2026-02-17T10:00:00Z",
                    "tool_name": "web_search",
                    "status": "success",
                    "execution_time_ms": 500,
                },
                {
                    "timestamp": "2026-02-17T10:01:00Z",
                    "tool_name": "http_get",
                    "status": "error",
                    "execution_time_ms": 3000,
                },
            ])
            report.add_page_break()
            report.add_section("Conclusion", "Done.")

            filepath = report.save()
            assert Path(filepath).exists()
            size = Path(filepath).stat().st_size
            assert size > 1000  # Non-trivial PDF

    def test_auto_filename(self):
        """save() without filename generates a timestamped name."""
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportBuilder(title="Auto Name Test", output_dir=tmpdir)
            filepath = report.save()
            filename = Path(filepath).name
            assert filename.startswith("Auto_Name_Test_")
            assert filename.endswith(".pdf")

    def test_custom_filename(self):
        """save() with explicit filename uses it."""
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportBuilder(title="Custom", output_dir=tmpdir)
            filepath = report.save("my_report.pdf")
            assert Path(filepath).name == "my_report.pdf"

    def test_output_dir_created(self):
        """ReportBuilder creates output directory if it doesn't exist."""
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "sub" / "dir"
            report = ReportBuilder(title="Nested", output_dir=str(nested))
            filepath = report.save()
            assert nested.exists()
            assert Path(filepath).exists()

    def test_empty_sections(self):
        """Report with no content sections still produces valid PDF."""
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportBuilder(title="Empty", output_dir=tmpdir)
            filepath = report.save()
            assert Path(filepath).exists()

    def test_memory_records_trust_levels(self):
        """Memory records render all three trust levels."""
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportBuilder(title="Trust Test", output_dir=tmpdir)
            report.add_memory_records([
                {"id": "high1", "content": "High trust", "trust_score": 0.9,
                 "source": "user", "tags": ["verified"]},
                {"id": "mid1", "content": "Mid trust", "trust_score": 0.5,
                 "source": "agent", "tags": []},
                {"id": "low1", "content": "Low trust", "trust_score": 0.1,
                 "source": "auto", "tags": ["unverified"]},
            ])
            filepath = report.save()
            assert Path(filepath).exists()

    def test_long_content_truncated(self):
        """Memory record content > 80 chars is truncated."""
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportBuilder(title="Trunc", output_dir=tmpdir)
            long_content = "A" * 200
            report.add_memory_records([
                {"id": "trunc1", "content": long_content, "trust_score": 0.5,
                 "source": "test", "tags": []},
            ])
            filepath = report.save()
            assert Path(filepath).exists()

    def test_report_body_is_written_beyond_cover_page(self):
        """Regression: body content should not disappear leaving only the cover page."""
        fitz = pytest.importorskip("fitz")
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportBuilder(
                title="Стратегічний план автономії та виживання: Березень 2026",
                output_dir=tmpdir,
            )
            report.add_section(
                "Резюме",
                "Це основний зміст звіту.\n\n- Пункт 1\n- Пункт 2\n1. Наступний крок",
            )
            filepath = report.save()

            doc = fitz.open(filepath)
            assert doc.page_count >= 2
            extracted = "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))
            doc.close()
            assert "Це основний зміст звіту" in extracted
            assert "Пункт 1" in extracted

    def test_financial_report_preamble_and_wrapped_table_cells(self):
        """Financial/VAT reports should include contents + summary preamble and keep long table text."""
        fitz = pytest.importorskip("fitz")
        from remy.core.report_builder import ReportBuilder

        with tempfile.TemporaryDirectory() as tmpdir:
            report = ReportBuilder(
                title="ПДВ звіт за березень 2026",
                output_dir=tmpdir,
                report_type="vat",
                include_toc=True,
                metadata={
                    "period": "Березень 2026",
                    "currency": "UAH",
                    "tax_id": "1234567890",
                },
            )
            report.add_section("Огляд", "Стислий опис документа.")
            report.add_table(
                title="Розрахунок",
                headers=["Показник", "Опис"],
                rows=[[
                    "ПДВ до сплати",
                    "Дуже довгий опис рядка для перевірки переносу в таблиці українською мовою без втрати тексту.",
                ]],
            )
            filepath = report.save()

            doc = fitz.open(filepath)
            extracted = "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))
            doc.close()

            assert "Contents" in extracted
            assert "Financial Document Summary" in extracted
            assert "Березень 2026" in extracted
            assert "Дуже довгий опис рядка" in extracted


# ============== GENERATE REPORT HELPER ==============


class TestGenerateReport:

    @patch("remy.core.notification_router.notify")
    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_success(self, mock_settings, mock_brain, mock_notify):
        """Successful report generation saves file and brain record."""
        from remy.core.brain_tools import _generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)

            mock_rec = MagicMock()
            mock_rec.id = "rpt-001"
            mock_brain.store.return_value = mock_rec

            result = _generate_report(
                {
                    "title": "Test Report",
                    "subtitle": "A test",
                    "sections": [
                        {"type": "section", "title": "Overview", "body": "Hello world"},
                        {"type": "findings", "title": "Results", "items": ["A", "B"]},
                    ],
                },
                session_id="test-session",
                channel="desktop",
            )

            parsed = json.loads(result)
            assert parsed["generated"] is True
            assert parsed["title"] == "Test Report"
            assert parsed["verification"]["status"] == "verified"
            assert parsed["verification"]["verified"] is True
            assert "filename" in parsed
            assert parsed["url"].startswith("/api/reports/")
            assert parsed["record_id"] == "rpt-001"

            # PDF file should exist on disk
            report_dir = Path(tmpdir) / "reports"
            files = list(report_dir.glob("*.pdf"))
            assert len(files) == 1

            # Brain record should be stored
            mock_brain.store.assert_called_once()
            call_kwargs = mock_brain.store.call_args
            assert "generated-report" in call_kwargs.kwargs.get(
                "tags", call_kwargs[1].get("tags", [])
            )
            mock_notify.assert_called_once()
            assert mock_notify.call_args.kwargs["event_type"] == "verification.resolved"

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_empty_sections(self, mock_settings, mock_brain):
        """Report with empty sections list still works."""
        from remy.core.brain_tools import _generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)

            mock_rec = MagicMock()
            mock_rec.id = "rpt-002"
            mock_brain.store.return_value = mock_rec

            result = _generate_report(
                {"title": "Empty Report", "sections": []},
                session_id=None,
                channel=None,
            )

            parsed = json.loads(result)
            assert parsed["generated"] is True

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_all_section_types(self, mock_settings, mock_brain):
        """Report with all section types generates correctly."""
        from remy.core.brain_tools import _generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)

            mock_rec = MagicMock()
            mock_rec.id = "rpt-003"
            mock_brain.store.return_value = mock_rec

            result = _generate_report(
                {
                    "title": "Full Report",
                    "sections": [
                        {"type": "section", "title": "S1", "body": "text"},
                        {"type": "subsection", "title": "Sub1", "body": "sub text"},
                        {"type": "text", "body": "plain text"},
                        {"type": "quote", "body": "a quote"},
                        {"type": "findings", "title": "Findings", "items": ["F1"]},
                        {"type": "table", "title": "T1",
                         "headers": ["H1", "H2"], "rows": [["a", "b"]]},
                        {"type": "page_break"},
                        {"type": "section", "title": "End", "body": "done"},
                    ],
                },
                session_id=None,
                channel=None,
            )

            parsed = json.loads(result)
            assert parsed["generated"] is True

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_sections_with_content_and_no_type_still_render_body(self, mock_settings, mock_brain):
        """LLM often sends title+content without type/body; PDF should still contain body text."""
        fitz = pytest.importorskip("fitz")
        from remy.core.brain_tools import _generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)

            mock_rec = MagicMock()
            mock_rec.id = "rpt-004"
            mock_brain.store.return_value = mock_rec

            result = _generate_report(
                {
                    "title": "Стратегічний план автономії та виживання: Березень 2026",
                    "sections": [
                        {
                            "title": "Короткий висновок",
                            "content": "Це детальний зміст звіту про ПДВ та фінансовий план.",
                        },
                        {
                            "title": "Наступні дії",
                            "content": "- Підготувати таблицю\n- Звірити платежі",
                        },
                    ],
                },
                session_id="test-session",
                channel="desktop",
            )

            parsed = json.loads(result)
            pdf_path = Path(tmpdir) / "reports" / parsed["filename"]
            assert pdf_path.exists()

            doc = fitz.open(pdf_path)
            assert doc.page_count >= 2
            extracted = "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))
            doc.close()
            assert "Це детальний зміст звіту про ПДВ та фінансовий план." in extracted
            assert "Підготувати таблицю" in extracted

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_content_only_report_renders_full_body(self, mock_settings, mock_brain):
        """LLM may send title+content without sections; this should still generate a full PDF."""
        fitz = pytest.importorskip("fitz")
        from remy.core.brain_tools import _generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)

            mock_rec = MagicMock()
            mock_rec.id = "rpt-004b"
            mock_brain.store.return_value = mock_rec

            result = _generate_report(
                {
                    "title": "Стратегічний план AuraSDK: Релокація та Монетизація",
                    "content": """# СТРАТЕГІЧНИЙ ПЛАН: AURA SDK — МОНЕТИЗАЦІЯ ТА РЕЛОКАЦІЯ

1. ПРИНЦИП "ТЕХНОЛОГІЧНОГО ПРИВИДА"

* Жодних українських ФОП/ТОВ
* Shadow Monetization

2. МОДЕЛЬ OPEN CORE

* Aura Core (Public)
* Aura Enterprise (Private)
""",
                },
                session_id="test-session",
                channel="desktop",
            )

            parsed = json.loads(result)
            pdf_path = Path(tmpdir) / "reports" / parsed["filename"]
            assert pdf_path.exists()

            doc = fitz.open(pdf_path)
            assert doc.page_count >= 2
            extracted = "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))
            doc.close()
            assert "Жодних українських ФОП/ТОВ" in extracted
            assert "Aura Enterprise (Private)" in extracted

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    def test_financial_report_options_are_rendered(self, mock_settings, mock_brain):
        """generate_report should support TOC + financial metadata preamble."""
        fitz = pytest.importorskip("fitz")
        from remy.core.brain_tools import _generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)

            mock_rec = MagicMock()
            mock_rec.id = "rpt-005"
            mock_brain.store.return_value = mock_rec

            result = _generate_report(
                {
                    "title": "ПДВ звіт за березень 2026",
                    "report_type": "vat",
                    "include_toc": True,
                    "metadata": {
                        "period": "Березень 2026",
                        "currency": "UAH",
                    },
                    "sections": [
                        {"title": "Підсумок", "content": "Огляд ПДВ зобов'язань."},
                    ],
                },
                session_id="test-session",
                channel="desktop",
            )

            parsed = json.loads(result)
            pdf_path = Path(tmpdir) / "reports" / parsed["filename"]

            doc = fitz.open(pdf_path)
            extracted = "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))
            doc.close()

            assert "Contents" in extracted
            assert "Financial Document Summary" in extracted
            assert "Березень 2026" in extracted

    @patch("remy.core.brain_tools.brain")
    @patch("remy.core.brain_tools.settings")
    @patch("remy.core.notification_router.notify")
    @patch("remy.core.report_builder.verify_generated_report", return_value=(False, "PDF appears to contain only a cover page without report body."))
    def test_generate_report_returns_error_when_pdf_validation_fails(
        self,
        _mock_verify,
        mock_notify,
        mock_settings,
        mock_brain,
    ):
        """Tool should not claim success when generated PDF fails validation."""
        from remy.core.brain_tools import _generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_settings.DATA_DIR = Path(tmpdir)

            mock_rec = MagicMock()
            mock_rec.id = "rpt-006"
            mock_brain.store.return_value = mock_rec

            result = _generate_report(
                {
                    "title": "Broken Report",
                    "sections": [{"title": "Only Cover?", "content": "Body"}],
                },
                session_id="test-session",
                channel="desktop",
            )

            parsed = json.loads(result)
            assert parsed["generated"] is False
            assert "cover page" in parsed["error"].lower()
            assert parsed["verification"]["status"] == "repair_required"
            assert parsed["verification"]["verified"] is False
            assert parsed["verification"]["failure_code"] == "verification_failed"
            mock_notify.assert_called_once()
            assert parsed["verification"]["repair_required"] is True


# ============== API ENDPOINT ==============


class TestServeReport:

    def _make_client(self):
        from fastapi.testclient import TestClient
        from remy.web.api import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_serve_report(self):
        """Endpoint serves existing PDF report."""
        from remy.config.settings import settings

        report_dir = Path(settings.DATA_DIR) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        test_report = report_dir / "test_report_123.pdf"
        test_report.write_bytes(b"%PDF-1.4 test content " + b"\x00" * 50)
        try:
            client = self._make_client()
            resp = client.get("/api/reports/test_report_123.pdf")
            assert resp.status_code == 200
            assert "application/pdf" in resp.headers["content-type"]
        finally:
            test_report.unlink(missing_ok=True)

    def test_not_found(self):
        """Endpoint returns 404 for missing report."""
        client = self._make_client()
        resp = client.get("/api/reports/nonexistent_xyz.pdf")
        assert resp.status_code == 404

    def test_path_traversal_blocked(self):
        """Endpoint blocks path traversal attempts."""
        client = self._make_client()
        resp = client.get("/api/reports/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (404, 422)


# ============== TELEGRAM REPORT DETECTION ==============


def test_telegram_report_regex():
    """Regex correctly finds report URL in response text."""
    import re

    response = "Here's your report: [Download Report](/api/reports/Test_Report_20260217_1200.pdf)"
    match = re.search(r'/api/reports/([\w.]+\.pdf)', response)
    assert match is not None
    assert match.group(1) == "Test_Report_20260217_1200.pdf"


def test_telegram_no_report_in_text():
    """Regex returns None when no report URL present."""
    import re

    response = "Hello! How can I help you today?"
    match = re.search(r'/api/reports/([\w.]+\.pdf)', response)
    assert match is None
