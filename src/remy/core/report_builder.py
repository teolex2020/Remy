"""
PDF Report Generator for Remy Agent.

Generates professional PDF reports from structured data.
Uses reportlab for PDF generation with full Unicode/Cyrillic support.
"""

import html
import json
import logging
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    HRFlowable,
)

logger = logging.getLogger(__name__)


# ============================================================
# FONT REGISTRATION (Cyrillic + Latin)
# ============================================================

_FONT_FAMILY = "Helvetica"  # fallback if no TTF found
_FONT_FAMILY_BOLD = "Helvetica-Bold"
_FONT_FAMILY_ITALIC = "Helvetica-Oblique"


def _register_unicode_fonts():
    """Register a TTF font family with Cyrillic support.

    Searches for Noto Sans, DejaVu Sans, or Arial on the system.
    Falls back to Helvetica (no Cyrillic) if nothing found.
    """
    global _FONT_FAMILY, _FONT_FAMILY_BOLD, _FONT_FAMILY_ITALIC

    # Candidate font families: (name, regular, bold, italic)
    system = platform.system()
    if system == "Windows":
        font_dir = Path("C:/Windows/Fonts")
    elif system == "Darwin":
        font_dir = Path("/Library/Fonts")
    else:
        font_dir = Path("/usr/share/fonts")

    candidates = [
        # Noto Sans — best Unicode coverage
        ("NotoSans", "NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf"),
        # DejaVu Sans — common on Linux
        ("DejaVuSans", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"),
        # Arial — Windows fallback
        ("Arial", "arial.ttf", "arialbd.ttf", "ariali.ttf"),
    ]

    for family_name, regular, bold, italic in candidates:
        # Search in font_dir and subdirectories
        reg_path = None
        for p in [font_dir / regular, *font_dir.rglob(regular)]:
            if p.exists():
                reg_path = p
                break

        if not reg_path:
            continue

        bold_path = reg_path.parent / bold
        italic_path = reg_path.parent / italic

        try:
            pdfmetrics.registerFont(TTFont(family_name, str(reg_path)))
            _FONT_FAMILY = family_name

            if bold_path.exists():
                bold_name = f"{family_name}-Bold"
                pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
                _FONT_FAMILY_BOLD = bold_name
            else:
                _FONT_FAMILY_BOLD = family_name

            if italic_path.exists():
                italic_name = f"{family_name}-Italic"
                pdfmetrics.registerFont(TTFont(italic_name, str(italic_path)))
                _FONT_FAMILY_ITALIC = italic_name
            else:
                _FONT_FAMILY_ITALIC = family_name

            logger.info("PDF fonts registered: %s from %s", family_name, reg_path.parent)
            return
        except Exception as e:
            logger.warning("Failed to register font %s: %s", family_name, e)

    logger.warning("No Unicode TTF fonts found — PDF Cyrillic will not render correctly")


_register_unicode_fonts()


# ============================================================
# COLOR SCHEME
# ============================================================

COLORS = {
    "primary": HexColor("#1a1a2e"),
    "secondary": HexColor("#16213e"),
    "accent": HexColor("#0f3460"),
    "highlight": HexColor("#e94560"),
    "text": HexColor("#2d2d2d"),
    "text_light": HexColor("#666666"),
    "bg_light": HexColor("#f5f5f5"),
    "bg_header": HexColor("#1a1a2e"),
    "white": HexColor("#ffffff"),
    "trust_high": HexColor("#27ae60"),
    "trust_mid": HexColor("#f39c12"),
    "trust_low": HexColor("#e74c3c"),
}


def _build_report_paragraphs(text: str) -> list[str]:
    """Convert plain/markdown-like text into ReportLab-safe paragraph fragments."""
    if not text:
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    paragraphs: list[str] = []
    blocks = re.split(r"\n\s*\n", normalized)
    for block in blocks:
        lines = []
        for raw_line in block.split("\n"):
            stripped = raw_line.strip()
            if not stripped:
                continue

            bullet_match = re.match(r"^[-*]\s+(.+)$", stripped)
            if bullet_match:
                lines.append(f"&#8226; {html.escape(bullet_match.group(1).strip())}")
                continue

            number_match = re.match(r"^(\d+)[\.\)]\s+(.+)$", stripped)
            if number_match:
                lines.append(
                    f"<b>{html.escape(number_match.group(1))}.</b> "
                    f"{html.escape(number_match.group(2).strip())}"
                )
                continue

            lines.append(html.escape(stripped))

        if lines:
            paragraphs.append("<br/>".join(lines))

    return paragraphs


def verify_generated_report(filepath: str | Path, *, title: str = "") -> tuple[bool, str]:
    """Best-effort sanity check that a generated PDF contains body content."""
    pdf_path = Path(filepath)
    if not pdf_path.exists() or pdf_path.stat().st_size < 1024:
        return False, "PDF file is missing or unexpectedly small."

    try:
        import fitz
    except Exception:
        return True, "PyMuPDF not installed; skipped deep PDF validation."

    doc = None
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count < 1:
            return False, "PDF has no pages."

        extracted = "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count)).strip()
        if len(extracted) < 80:
            return False, "PDF text extraction is nearly empty."

        normalized = extracted.lower()
        title_lower = (title or "").strip().lower()

        if doc.page_count == 1 and title_lower and normalized.count(title_lower) >= 1 and len(extracted) < 500:
            return False, "PDF appears to contain only a cover page without report body."

        return True, "PDF content looks valid."
    except Exception as e:
        logger.warning("PDF verification failed for %s: %s", pdf_path, e)
        return True, f"Skipped deep PDF verification due to parser error: {e}"
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def get_report_styles():
    """Create report paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=24,
        textColor=COLORS["primary"],
        spaceAfter=6 * mm,
        alignment=TA_CENTER,
        fontName=_FONT_FAMILY_BOLD,
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=COLORS["text_light"],
        spaceAfter=10 * mm,
        alignment=TA_CENTER,
        fontName=_FONT_FAMILY,
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=COLORS["primary"],
        spaceBefore=8 * mm,
        spaceAfter=4 * mm,
        fontName=_FONT_FAMILY_BOLD,
        borderWidth=0,
        borderPadding=0,
    ))

    styles.add(ParagraphStyle(
        name="SubSectionTitle",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=COLORS["secondary"],
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
        fontName=_FONT_FAMILY_BOLD,
    ))

    styles.add(ParagraphStyle(
        name="ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        textColor=COLORS["text"],
        spaceAfter=3 * mm,
        alignment=TA_JUSTIFY,
        fontName=_FONT_FAMILY,
        leading=14,
    ))

    styles.add(ParagraphStyle(
        name="ReportQuote",
        parent=styles["Normal"],
        fontSize=10,
        textColor=COLORS["accent"],
        spaceAfter=3 * mm,
        leftIndent=15 * mm,
        rightIndent=10 * mm,
        fontName=_FONT_FAMILY_ITALIC,
        leading=14,
        borderWidth=1,
        borderColor=COLORS["accent"],
        borderPadding=5,
    ))

    styles.add(ParagraphStyle(
        name="MetaInfo",
        parent=styles["Normal"],
        fontSize=9,
        textColor=COLORS["text_light"],
        alignment=TA_CENTER,
        fontName=_FONT_FAMILY,
    ))

    styles.add(ParagraphStyle(
        name="TrustLabel",
        parent=styles["Normal"],
        fontSize=8,
        textColor=COLORS["text_light"],
        fontName=_FONT_FAMILY,
    ))

    styles.add(ParagraphStyle(
        name="TOCTitle",
        parent=styles["Heading1"],
        fontSize=17,
        textColor=COLORS["primary"],
        spaceBefore=4 * mm,
        spaceAfter=5 * mm,
        fontName=_FONT_FAMILY_BOLD,
    ))

    styles.add(ParagraphStyle(
        name="TOCEntry",
        parent=styles["Normal"],
        fontSize=10,
        textColor=COLORS["text"],
        spaceAfter=2 * mm,
        fontName=_FONT_FAMILY,
        leading=13,
    ))

    styles.add(ParagraphStyle(
        name="TOCSubEntry",
        parent=styles["TOCEntry"],
        leftIndent=6 * mm,
        textColor=COLORS["secondary"],
    ))

    styles.add(ParagraphStyle(
        name="TableHeaderCell",
        parent=styles["Normal"],
        fontSize=9,
        textColor=COLORS["white"],
        fontName=_FONT_FAMILY_BOLD,
        alignment=TA_CENTER,
        leading=11,
    ))

    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=COLORS["text"],
        fontName=_FONT_FAMILY,
        leading=11,
        wordWrap="CJK",
    ))

    return styles


# ============================================================
# HEADER / FOOTER
# ============================================================

class ReportTemplate:
    """Header and footer for every page."""

    def __init__(self, title: str, author: str = "Remy AI Agent"):
        self.title = title
        self.author = author

    def header_footer(self, canvas_obj, doc):
        canvas_obj.saveState()
        width, height = A4

        # Header line
        canvas_obj.setStrokeColor(COLORS["accent"])
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(20 * mm, height - 15 * mm, width - 20 * mm, height - 15 * mm)

        canvas_obj.setFont(_FONT_FAMILY, 8)
        canvas_obj.setFillColor(COLORS["text_light"])
        canvas_obj.drawString(20 * mm, height - 13 * mm, self.title)
        canvas_obj.drawRightString(width - 20 * mm, height - 13 * mm, self.author)

        # Footer
        canvas_obj.setStrokeColor(COLORS["accent"])
        canvas_obj.line(20 * mm, 15 * mm, width - 20 * mm, 15 * mm)

        canvas_obj.setFont(_FONT_FAMILY, 8)
        canvas_obj.setFillColor(COLORS["text_light"])
        canvas_obj.drawCentredString(width / 2, 10 * mm, f"Page {doc.page}")
        canvas_obj.drawString(
            20 * mm, 10 * mm,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        )

        canvas_obj.restoreState()


# ============================================================
# REPORT BUILDER
# ============================================================

class ReportBuilder:
    """
    Builds a PDF report step-by-step.

    Usage:
        report = ReportBuilder("Research Report", "AI Agent Analysis")
        report.add_section("Introduction", "This report covers...")
        report.add_key_findings(["Finding 1", "Finding 2"])
        report.add_table(headers, rows)
        report.add_memory_records(records)
        report.save("report.pdf")
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        author: str = "Remy AI Agent",
        output_dir: Optional[str] = None,
        report_type: str = "standard",
        include_toc: bool = True,
        metadata: Optional[dict] = None,
    ):
        self.title = title
        self.report_type = report_type or "standard"
        if not subtitle and self.report_type in {"financial", "vat"}:
            subtitle = "Financial / VAT Report"
        self.subtitle = subtitle
        self.author = author
        self.include_toc = include_toc
        self.metadata = metadata or {}
        self.output_dir = Path(output_dir or "./data/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = get_report_styles()
        self.story = []
        self._cover_len = 0
        self._toc_entries: list[tuple[int, str]] = []
        self._build_cover()

    def _build_cover(self):
        """Cover page."""
        self.story.append(Spacer(1, 40 * mm))
        self.story.append(Paragraph(self.title, self.styles["ReportTitle"]))

        if self.subtitle:
            self.story.append(Paragraph(self.subtitle, self.styles["ReportSubtitle"]))

        self.story.append(HRFlowable(
            width="60%", thickness=1, color=COLORS["accent"],
            spaceBefore=5 * mm, spaceAfter=5 * mm, hAlign="CENTER",
        ))

        now = datetime.now().strftime("%d %B %Y, %H:%M")
        self.story.append(Paragraph(
            f"Author: {self.author}<br/>Date: {now}",
            self.styles["MetaInfo"],
        ))
        self.story.append(PageBreak())
        self._cover_len = len(self.story)

    def _build_preamble_story(self) -> list:
        """Build TOC and document metadata pages inserted after the cover."""
        story = []

        if self.include_toc and self._toc_entries:
            story.append(Paragraph("Contents", self.styles["TOCTitle"]))
            for level, title in self._toc_entries:
                style = self.styles["TOCSubEntry"] if level > 1 else self.styles["TOCEntry"]
                story.append(Paragraph(html.escape(title), style))
            story.append(PageBreak())

        if self.metadata:
            title = "Document Summary" if self.report_type not in {"financial", "vat"} else "Financial Document Summary"
            story.append(Paragraph(title, self.styles["TOCTitle"]))
            rows = []
            for key, value in self.metadata.items():
                if value is None or value == "":
                    continue
                label = str(key).replace("_", " ").strip().title()
                rows.append([
                    Paragraph(f"<b>{html.escape(label)}</b>", self.styles["TableCell"]),
                    Paragraph(html.escape(str(value)), self.styles["TableCell"]),
                ])

            if rows:
                available_width = A4[0] - 40 * mm
                table = Table(rows, colWidths=[available_width * 0.28, available_width * 0.72])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), COLORS["bg_light"]),
                    ("GRID", (0, 0), (-1, -1), 0.5, COLORS["text_light"]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(table)
                story.append(PageBreak())

        return story

    def add_section(self, title: str, body: str = ""):
        """Add a section with heading and text."""
        if title:
            self._toc_entries.append((1, title))
        self.story.append(Paragraph(html.escape(title), self.styles["SectionTitle"]))
        self.story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=COLORS["accent"], spaceAfter=3 * mm,
        ))
        if body:
            for paragraph in _build_report_paragraphs(body):
                self.story.append(Paragraph(paragraph, self.styles["ReportBody"]))

    def add_subsection(self, title: str, body: str = ""):
        """Add a subsection."""
        if title:
            self._toc_entries.append((2, title))
        self.story.append(Paragraph(html.escape(title), self.styles["SubSectionTitle"]))
        if body:
            for paragraph in _build_report_paragraphs(body):
                self.story.append(Paragraph(paragraph, self.styles["ReportBody"]))

    def add_text(self, text: str):
        """Add body text."""
        for paragraph in _build_report_paragraphs(text):
            self.story.append(Paragraph(paragraph, self.styles["ReportBody"]))

    def add_quote(self, text: str):
        """Add a styled quote block."""
        for paragraph in _build_report_paragraphs(text):
            self.story.append(Paragraph(paragraph, self.styles["ReportQuote"]))

    def add_key_findings(self, findings: list[str], title: str = "Key Findings"):
        """Add a numbered findings block."""
        self.story.append(Paragraph(html.escape(title), self.styles["SectionTitle"]))
        self.story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=COLORS["accent"], spaceAfter=3 * mm,
        ))
        for i, finding in enumerate(findings, 1):
            self.story.append(Paragraph(
                f"<b>{i}.</b> {html.escape(str(finding))}", self.styles["ReportBody"],
            ))
            self.story.append(Spacer(1, 2 * mm))

    def add_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        title: str = "",
        col_widths: Optional[list[float]] = None,
    ):
        """Add a styled data table."""
        if title:
            self.story.append(Paragraph(html.escape(title), self.styles["SubSectionTitle"]))

        if not headers:
            return

        table_data = [[Paragraph(html.escape(str(cell)), self.styles["TableHeaderCell"]) for cell in headers]]
        for row in rows:
            padded = list(row) + [""] * max(0, len(headers) - len(row))
            table_data.append([
                Paragraph(html.escape(str(cell)), self.styles["TableCell"])
                for cell in padded[:len(headers)]
            ])

        if not col_widths:
            available_width = A4[0] - 40 * mm
            col_widths = [available_width / len(headers)] * len(headers)

        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["bg_header"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
            ("FONTNAME", (0, 0), (-1, 0), _FONT_FAMILY_BOLD),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("FONTNAME", (0, 1), (-1, -1), _FONT_FAMILY),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("TEXTCOLOR", (0, 1), (-1, -1), COLORS["text"]),
            ("ALIGN", (0, 1), (-1, -1), "LEFT"),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            *[
                ("BACKGROUND", (0, i), (-1, i), COLORS["bg_light"])
                for i in range(2, len(table_data), 2)
            ],
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["text_light"]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 5 * mm))

    def add_memory_records(
        self,
        records: list[dict],
        title: str = "Memory Records Analysis",
    ):
        """Add memory records table with trust indicators."""
        self.story.append(Paragraph(html.escape(title), self.styles["SectionTitle"]))
        self.story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=COLORS["accent"], spaceAfter=3 * mm,
        ))

        headers = ["ID", "Content", "Trust", "Source", "Tags"]
        rows = []

        for rec in records:
            trust = rec.get("trust_score", 0.5)
            if trust >= 0.7:
                trust_str = (
                    f'<font color="#{COLORS["trust_high"].hexval()[2:]}">'
                    f'{trust:.1f} &#x2713;</font>'
                )
            elif trust >= 0.4:
                trust_str = (
                    f'<font color="#{COLORS["trust_mid"].hexval()[2:]}">'
                    f'{trust:.1f} &#x25CB;</font>'
                )
            else:
                trust_str = (
                    f'<font color="#{COLORS["trust_low"].hexval()[2:]}">'
                    f'{trust:.1f} &#x2717;</font>'
                )

            content = rec.get("content", "")
            if len(content) > 80:
                content = content[:77] + "..."

            rows.append([
                rec.get("id", "")[:8],
                content,
                trust_str,
                rec.get("source", "unknown"),
                ", ".join(rec.get("tags", [])),
            ])

        w = A4[0] - 40 * mm
        col_widths = [w * 0.10, w * 0.40, w * 0.10, w * 0.18, w * 0.22]

        table_data = [headers]
        for row in rows:
            table_data.append([
                Paragraph(html.escape(row[0]), self.styles["TableCell"]),
                Paragraph(html.escape(row[1]), self.styles["TableCell"]),
                Paragraph(row[2], self.styles["TrustLabel"]),
                Paragraph(html.escape(row[3]), self.styles["TableCell"]),
                Paragraph(html.escape(row[4]), self.styles["TableCell"]),
            ])

        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["bg_header"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
            ("FONTNAME", (0, 0), (-1, 0), _FONT_FAMILY_BOLD),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTNAME", (0, 1), (-1, -1), _FONT_FAMILY),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("TEXTCOLOR", (0, 1), (-1, -1), COLORS["text"]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            *[
                ("BACKGROUND", (0, i), (-1, i), COLORS["bg_light"])
                for i in range(2, len(table_data), 2)
            ],
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["text_light"]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 5 * mm))

    def add_audit_summary(
        self,
        audit_logs: list[dict],
        title: str = "Execution Audit Trail",
    ):
        """Add audit trail summary with statistics."""
        self.story.append(Paragraph(html.escape(title), self.styles["SectionTitle"]))
        self.story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=COLORS["accent"], spaceAfter=3 * mm,
        ))

        total = len(audit_logs)
        success = sum(1 for l in audit_logs if l.get("status") == "success")
        errors = sum(1 for l in audit_logs if l.get("status") == "error")
        timeouts = sum(1 for l in audit_logs if l.get("status") == "timeout")

        stats_text = (
            f"Total actions: <b>{total}</b> | "
            f'Success: <b><font color="#27ae60">{success}</font></b> | '
            f'Errors: <b><font color="#e74c3c">{errors}</font></b> | '
            f'Timeouts: <b><font color="#f39c12">{timeouts}</font></b>'
        )
        self.story.append(Paragraph(stats_text, self.styles["ReportBody"]))
        self.story.append(Spacer(1, 3 * mm))

        if audit_logs:
            headers = ["Time", "Tool", "Status", "Duration"]
            rows = []
            for log in audit_logs[:20]:
                ts = log.get("timestamp", "")[:19].replace("T", " ")
                status = log.get("status", "unknown")
                status_icon = {
                    "success": "OK", "error": "FAIL", "timeout": "TIMEOUT",
                }.get(status, "?")
                duration = f"{log.get('execution_time_ms', 0):.0f}ms"
                rows.append([ts, log.get("tool_name", ""), status_icon, duration])
            self.add_table(headers, rows)

    def add_page_break(self):
        """Insert a page break."""
        self.story.append(PageBreak())

    def save(self, filename: Optional[str] = None) -> str:
        """Save PDF and return the filepath."""
        if not filename:
            safe_title = "".join(
                c if c.isalnum() or c in " -_" else "_"
                for c in self.title
            ).strip().replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_title}_{timestamp}.pdf"

        filepath = self.output_dir / filename
        template = ReportTemplate(self.title, self.author)

        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
        )

        preamble = self._build_preamble_story()
        story = self.story[:self._cover_len] + preamble + self.story[self._cover_len:]

        doc.build(
            story,
            onFirstPage=template.header_footer,
            onLaterPages=template.header_footer,
        )

        return str(filepath)
