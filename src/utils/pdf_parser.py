import fitz  # PyMuPDF
from typing import Dict, Any, List
import os


# clean lines by stripping whitespace and removing empties
def _clean_lines(lines: List[str]) -> List[str]:

    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned.append(line)
    return cleaned


# Title extraction
def _extract_title(clean_lines: List[str], page_idx: int) -> str:

    if not clean_lines:
        return f"Slide {page_idx + 1}"

    # Case 1: first line is reasonably short → likely the slide title
    first = clean_lines[0]
    if len(first.split()) <= 12:
        return first

    # Case 2: pick the shortest non-empty line
    shortest = min(clean_lines, key=lambda l: len(l))
    if len(shortest.split()) <= 12:
        return shortest

    # Fallback
    return f"Slide {page_idx + 1}"


# Diagram detection heuristic
def _detect_diagram(text: str) -> bool:
    words = text.split()
    return len(words) < 15


# Main PDF to slides extraction
def extract_slides_from_pdf(pdf_path: str) -> Dict[str, Any]:

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    slides = []

    for page_idx, page in enumerate(doc):
        raw_text = page.get_text("text")

        # Split into lines & clean
        raw_lines = raw_text.split("\n")
        clean_lines = _clean_lines(raw_lines)

        # If the page is extremely empty → substitute placeholder
        if not clean_lines:
            clean_lines = ["(No readable text extracted from this slide.)"]

        # Title
        title = _extract_title(clean_lines, page_idx)

        # Content = everything except title
        body_lines = clean_lines[1:] if len(clean_lines) > 1 else []
        content = "\n".join(body_lines).strip()

        # Guarantee non-empty content
        if not content:
            content = "[This slide contains mainly visual content.]"

        # Diagram detection
        if _detect_diagram(content):
            content += "\n[Diagram detected — summarize conceptual meaning rather than visuals.]"

        slides.append({
            "title": title,
            "content": content,
            "raw_text": raw_text,
            "page_number": page_idx + 1,
        })

    lecture_title = os.path.basename(pdf_path).replace(".pdf", "")

    return {
        "lecture_title": lecture_title,
        "slides": slides
    }
