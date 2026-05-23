import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            if text:
                full_text += text + "\n"
    return full_text