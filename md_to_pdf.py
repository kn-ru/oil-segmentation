"""
Конвертация Markdown-отчёта в PDF с встроенными графиками.
Использует markdown + weasyprint.
"""

import os
import markdown
from weasyprint import HTML

REPORT_DIR = "/home/knru/EcoAir/oil_detection/report"
MD_PATH = os.path.join(REPORT_DIR, "report.md")
PDF_PATH = os.path.join(REPORT_DIR, "report.pdf")

CSS = """
@page {
    size: A4;
    margin: 2cm;
}
body {
    font-family: "DejaVu Sans", "Noto Sans", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
}
h1 {
    font-size: 20pt;
    color: #1a3c5e;
    border-bottom: 2px solid #1a3c5e;
    padding-bottom: 6px;
    margin-top: 30px;
}
h2 {
    font-size: 15pt;
    color: #2a5f8f;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    margin-top: 24px;
}
h3 {
    font-size: 12pt;
    color: #3a7fbf;
    margin-top: 18px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 10pt;
}
th, td {
    border: 1px solid #bbb;
    padding: 6px 10px;
    text-align: left;
}
th {
    background-color: #e8f0f8;
    font-weight: bold;
}
tr:nth-child(even) {
    background-color: #f9f9f9;
}
img {
    max-width: 100%;
    display: block;
    margin: 16px auto;
}
blockquote {
    border-left: 4px solid #2a5f8f;
    margin: 12px 0;
    padding: 8px 16px;
    background-color: #f0f6fc;
    font-style: italic;
}
em {
    color: #555;
    font-size: 9.5pt;
}
code {
    background-color: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 10pt;
}
"""


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    # Оборачиваем в полный HTML с base_url для разрешения относительных путей к картинкам
    html_full = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""

    print("Генерация PDF...")
    HTML(string=html_full, base_url=REPORT_DIR).write_pdf(PDF_PATH)
    size_mb = os.path.getsize(PDF_PATH) / 1024 / 1024
    print(f"Готово: {PDF_PATH} ({size_mb:.1f} МБ)")


if __name__ == "__main__":
    main()
