from fpdf import FPDF


def save_tabs_to_pdf(tabs: str, output_path: str):
    """
    Saves guitar tabs to a PDF file.

    Args:
        tabs (str): Guitar tabs as a string.
        output_path (str): Path to save the PDF file.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Courier", size=12)

    # Add each line of the tabs to the PDF
    for line in tabs.split("\n"):
        pdf.cell(0, 10, line, ln=True)

    pdf.output(output_path)
    print(f"Tabs saved to {output_path}")
