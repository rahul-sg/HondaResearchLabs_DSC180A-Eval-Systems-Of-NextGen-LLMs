from pathlib import Path
from typing import Dict, Any, List

def extract_slides_from_txt(path: str) -> Dict[str, Any]:
    slides: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

    for i, block in enumerate(blocks):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        title = lines[0]
        bullets = lines[1:] if len(lines) > 1 else []

        slides.append({
            "slide_id": i,
            "title": title,
            "bullets": bullets
        })

    return {
        "source": path,
        "format": "txt",
        "slides": slides
    }
