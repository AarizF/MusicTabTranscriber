from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

from .models import ConfidenceSummary, TabEvent
from .tab_generator import render_ascii_tab


def save_tabs_to_pdf(tab_events: list[TabEvent], summary: ConfidenceSummary, output_path: str) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=11)
    pdf.cell(0, 10, "MusicTabTranscriber Result", ln=True)
    pdf.cell(0, 8, f"Notes: {summary.note_count}", ln=True)
    pdf.cell(0, 8, f"Average confidence: {summary.average_confidence:.2f}", ln=True)
    pdf.cell(0, 8, f"Branch used: {summary.branch_used}", ln=True)
    pdf.ln(4)

    for line in render_ascii_tab(tab_events).splitlines():
        pdf.multi_cell(0, 7, line)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(output_path)
