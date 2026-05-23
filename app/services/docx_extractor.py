from docx import Document

def extract_text_from_docx(docx_path):

    doc = Document(docx_path)

    full_text = ""

    for para in doc.paragraphs:
        full_text += para.text + "\n"

    return full_text