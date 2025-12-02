import fitz  # PyMuPDF

def read_pdf(pdf_path):
    #fitz-library extracts text from PDF pages reliably.
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
