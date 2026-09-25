
import re


def _find_chunk_end(text, start, chunk_size):
    """Find a sentence or word boundary near the chunk-size limit."""

    limit = min(start + chunk_size, len(text))

    if limit == len(text):
        return limit

    window = text[start:limit]

    # Prefer the last sentence boundary within the chunk.
    sentence_ends = list(re.finditer(r"[.!?](?:\s+|$)", window))

    if sentence_ends:
        candidate = start + sentence_ends[-1].end()
        if candidate - start >= chunk_size // 2:
            return candidate

    # Otherwise, end at the last complete word.
    last_space = window.rfind(" ")

    if last_space >= chunk_size // 2:
        return start + last_space + 1

    # A very long word or sentence may require a hard boundary.
    return limit


def _find_next_start(text, previous_start, end, overlap):
    """Move back approximately overlap characters to a word boundary."""

    if end >= len(text):
        return end

    target = max(previous_start + 1, end - overlap)

    # Move forward to the next word boundary so the next chunk
    # does not start in the middle of a word.
    next_space = text.find(" ", target, end)

    if next_space != -1:
        target = next_space + 1

    return min(target, end)


def create_chunks(pages, document_name, chunk_size=1000, overlap=200):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    chunk_id = 0

    for page in pages:
        text = re.sub(r"\s+", " ", page["text"]).strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = _find_chunk_end(text, start, chunk_size)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "document": document_name,
                    "page": page["page"],
                    "text": chunk_text
                })
                chunk_id += 1

            if end >= len(text):
                break

            next_start = _find_next_start(text, start, end, overlap)

            # Guarantee forward progress.
            if next_start <= start:
                next_start = end

            start = next_start

    return chunks