import os
import json
from typing import Dict, Any
from pathlib import Path

from src.utils.pdf_parser import extract_slides_from_pdf


# Loading slides from PDF/JSON
def load_slides(path: str) -> Dict[str, Any]:
   
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Slide file not found: {path}")

    if path.suffix.lower() == ".pdf":
        return extract_slides_from_pdf(str(path))

    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported slide format: {path.suffix}")



# Write iterations and save
def write_iteration_summary(output_dir: str, iteration: int, text: str):

    if text is None:
        text = ""

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"iter_{iteration}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

# Final summary
def write_final_summary(output_dir: str, text: str):

    if text is None:
        text = ""

    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / "final.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")



# Final Results JSON
def write_json(path: str, data: Dict[str, Any]):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
