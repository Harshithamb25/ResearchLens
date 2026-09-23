from ingestion.ingest import ingest_all_papers


results = ingest_all_papers()

print("\n==============================")
print("RESEARCHLENS INGESTION")
print("==============================")

if not results:
    print("No papers were ingested.")
else:
    for result in results:
        print(f"\nDocument: {result['document']}")
        print(f"Pages: {result['pages']}")
        print(f"Chunks: {result['chunks']}")

    print("\nTotal papers:", len(results))