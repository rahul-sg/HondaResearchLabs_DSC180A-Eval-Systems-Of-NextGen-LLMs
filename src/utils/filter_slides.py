from typing import List, Dict
import re

# Check if slide is syllabus
def is_syllabus_slide(slide: Dict) -> bool:
    text = (slide.get("title", "") + " " + slide.get("content", "")).lower()

    # Common syllabus indicators
    syllabus_keywords = [
        "syllabus",
        "instructor",
        "office hours",
        "course info",
        "course information",
        "grading",
        "required materials",
        "policies",
        "academic integrity",
        "late work",
        "makeup",
        "assessment",
        "quiz",
        "final exam",
        "midterm",
        "ta:", "teaching assistant",
        "section",
        "ucsd",
        "university policy",
        "zoom",
        "canvas",
        "connect",
        "hbs case",
        "week 1–10",
        "schedule",
        "course overview",
        "learning objectives",
        "achieving the objective",
    ]

    return any(keyword in text for keyword in syllabus_keywords)

# Keep only non-syllabus slides
def filter_content_slides(slides: List[Dict]) -> List[Dict]:

    clean = [s for s in slides if not is_syllabus_slide(s)]
    return clean if len(clean) > 0 else slides
