from ingestion.pdf_loader import extract_text_from_pdf


pdf_path = "../data/papers/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

print("Total pages:", len(pages))

for page in pages:
    print(f"\n--- Page {page['page']} ---")
    print(page["text"][:500])