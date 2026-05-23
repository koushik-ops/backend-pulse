import os
import shutil
import pytesseract
from PIL import Image

# Dynamically locate Tesseract executable
tesseract_bin = shutil.which("tesseract")
if tesseract_bin:
    pytesseract.pytesseract.tesseract_cmd = tesseract_bin
else:
    # Try common Windows paths
    for path in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"D:\Tesseract-OCR\tesseract.exe",
    ]:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

def extract_text_from_image(image_path):
    try:
        # Check if tesseract command is actually set/callable
        if not pytesseract.pytesseract.tesseract_cmd or not os.path.exists(pytesseract.pytesseract.tesseract_cmd) and not shutil.which("tesseract"):
            raise FileNotFoundError("Tesseract binary not found on the system.")
            
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error during OCR extraction: {e}")
        raise RuntimeError(
            "Tesseract OCR engine is not installed or configured on this server. "
            "Please configure Tesseract or upload PDF/DOCX reports instead."
        )