"""
Simple Markdown -> PDF renderer using reportlab.

This script loads REPORT.md and renders it to REPORT.pdf in the repo root.
It performs basic markdown handling for headers and paragraphs.
"""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.colors import black

def render_markdown_to_pdf(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding='utf-8')
    lines = text.splitlines()

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    margin = 20 * mm
    x = margin
    y = height - margin

    # Basic fonts
    c.setFillColor(black)
    normal_font = ("Helvetica", 10)
    h1_font = ("Helvetica-Bold", 16)
    h2_font = ("Helvetica-Bold", 12)
    c.setFont(*normal_font)

    def newline(linespace=12):
        nonlocal y
        y -= linespace
        if y < margin:
            c.showPage()
            c.setFont(*normal_font)
            y = height - margin

    for raw in lines:
        line = raw.rstrip()
        if not line:
            newline(8)
            continue

        if line.startswith('=') and set(line.strip()) == {'='}:
            continue

        if line.startswith('# '):
            c.setFont(*h1_font)
            c.drawString(x, y, line[2:].strip())
            c.setFont(*normal_font)
            newline(18)
            continue

        if line.startswith('## '):
            c.setFont(*h2_font)
            c.drawString(x, y, line[3:].strip())
            c.setFont(*normal_font)
            newline(14)
            continue

        # Wrap long lines
        max_width = width - 2 * margin
        words = line.split(' ')
        current = ''
        for w in words:
            test = (current + ' ' + w).strip()
            if c.stringWidth(test, normal_font[0], normal_font[1]) > max_width:
                c.drawString(x, y, current)
                newline(12)
                current = w
            else:
                current = test
        if current:
            c.drawString(x, y, current)
            newline(12)

    c.save()


if __name__ == '__main__':
    md = Path('REPORT.md')
    pdf = Path('REPORT.pdf')
    if not md.exists():
        print('REPORT.md not found')
    else:
        try:
            render_markdown_to_pdf(md, pdf)
            print('Saved', pdf)
        except Exception as e:
            print('Error generating PDF:', e)
