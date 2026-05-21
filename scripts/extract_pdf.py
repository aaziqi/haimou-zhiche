
from pdfminer.high_level import extract_text
import sys

def read_pdf(file_path):
    try:
        text = extract_text(file_path)
        print(text)
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    pdf_path = r"d:\VScode\Graduation project\docs\附件2云南师范大学本科生毕业论文（设计）撰写基本规范.pdf"
    read_pdf(pdf_path)
