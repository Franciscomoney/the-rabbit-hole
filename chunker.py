import re
from typing import List

def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average)."""
    return len(text) // 4

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_chunk(text: str, target_tokens: int = 800, max_tokens: int = 1200) -> List[str]:
    """
    Split text into semantic chunks.

    - Tries to keep chunks around target_tokens
    - Respects sentence boundaries
    - Won't exceed max_tokens per chunk
    - Groups related sentences together
    """
    # Clean up the text
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        sentences = split_into_sentences(paragraph)

        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)

            # If single sentence exceeds max, split it
            if sentence_tokens > max_tokens:
                # Save current chunk if any
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split long sentence by commas or semicolons
                parts = re.split(r'[,;]', sentence)
                for part in parts:
                    part = part.strip()
                    if part:
                        part_tokens = estimate_tokens(part)
                        if current_tokens + part_tokens > max_tokens and current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = [part]
                            current_tokens = part_tokens
                        else:
                            current_chunk.append(part)
                            current_tokens += part_tokens
                continue

            # Check if adding this sentence would exceed target
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                # Save current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add paragraph break marker for context
        if current_chunk and current_tokens < target_tokens * 0.7:
            # Continue building chunk
            pass
        elif current_chunk:
            # Good stopping point at paragraph end
            if current_tokens >= target_tokens * 0.5:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF bytes."""
    import PyPDF2
    from io import BytesIO

    reader = PyPDF2.PdfReader(BytesIO(pdf_content))
    text_parts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)

    return '\n\n'.join(text_parts)

def process_book(content: bytes, filename: str) -> List[str]:
    """Process a book file and return semantic chunks."""
    if filename.lower().endswith('.pdf'):
        text = extract_text_from_pdf(content)
    elif filename.lower().endswith('.txt'):
        text = content.decode('utf-8', errors='ignore')
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    chunks = semantic_chunk(text)

    # Filter out very short chunks (likely noise)
    chunks = [c for c in chunks if len(c) > 50]

    return chunks
