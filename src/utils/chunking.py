from typing import List, Dict

# Estimate tokens and chunk slides accordingly
def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))

# Chunk slides into groups based on token count
def chunk_slides_by_tokens(
    slides: List[Dict],
    max_tokens: int = 1500,
    text_key: str = "content"
) -> List[List[Dict]]:

    chunks: List[List[Dict]] = []
    current, cur_tok = [], 0

    for s in slides:
        t = estimate_tokens(str(s.get(text_key, "")))

        # Start a new chunk if adding this slide would exceed limit
        if current and cur_tok + t > max_tokens:
            chunks.append(current)
            current, cur_tok = [], 0

        current.append(s)
        cur_tok += t

    if current:
        chunks.append(current)

    return chunks

# Convert slides to text with chunk & slide indices
def slides_to_text(
    slides: List[Dict],
    max_chunks: int = 3,
    max_tokens: int = 1500
) -> str:
    
    chunks = chunk_slides_by_tokens(slides, max_tokens=max_tokens)
    chunks = chunks[:max_chunks]

    out = []
    for ci, ch in enumerate(chunks, 1):
        for i, s in enumerate(ch, 1):
            title = s.get("title", "")
            content = s.get("content", "")
            out.append(f"[Chunk {ci} • Slide {i}] {title}\n{content}")

    return "\n\n".join(out)
