import fitz
# PyMuPDF lets ResearchLens extract text from research PDFs while preserving page-level information, which we'll later use for citations. 

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()
# output format
        pages.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return pages