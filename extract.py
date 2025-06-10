import fitz
import re
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return clean_text(text)

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_per_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = clean_text(page.get_text())
        pages.append(text)
    doc.close()
    return pages

def extract_text_from_multiple_pdfs(pdf_paths):
    all_text = []
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        all_text.append({
                "filename": os.path.basename(pdf_path),
                "text": text
            })
    return all_text

# loader.py
class PDFLoader:
    def __init__(self, path):
        self.path = path
        self.text = self._load_and_clean()

    def _load_and_clean(self):
        return extract_text_from_multiple_pdfs(self.path)

    def get_text(self):
        return self.text

    def get_pages(self):
        return extract_text_per_page(self.path)

if __name__ == "__main__":
    pdf_path = ["COSMOS.pdf", "s12559-022-10038-y.pdf"]  # Replace with your PDF file path
    loader = PDFLoader(pdf_path)    
    text  =loader.get_text()
    print("Extracted Text:\n", text[:1000])