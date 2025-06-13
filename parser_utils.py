import fitz  # PyMuPDF
import markdown
import os

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join(page.get_text() for page in doc)

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def parse_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    return markdown.markdown(md_content)

def parse_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return parse_pdf(file_path)
    elif ext == '.txt':
        return parse_txt(file_path)
    elif ext == '.md':
        return parse_md(file_path)
    else:
        raise ValueError("Unsupported file type")
