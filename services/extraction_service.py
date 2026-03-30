"""
Extraction Service – Phase 1
Uses regex patterns to extract specific biomarker values from raw OCR text.
Returns a structured dictionary of biomarker → value mappings.
"""

import re
from typing import Optional


# ─── Regex patterns for each biomarker ────────────────────────────
# Each pattern captures a numeric value (int or float) near the biomarker name.
BIOMARKER_PATTERNS: dict[str, list[re.Pattern]] = {
    "hemoglobin": [
        re.compile(r"(?:hemoglobin|hgb|hb)\s*[:\-]?\s*([\d]+\.?\d*)\s*(?:g/dL|g/dl|gm/dL)?", re.IGNORECASE),
    ],
    "glucose": [
        re.compile(r"(?:fasting\s+)?(?:blood\s+)?glucose\s*[:\-]?\s*([\d]+\.?\d*)\s*(?:mg/dL|mg/dl)?", re.IGNORECASE),
        re.compile(r"(?:FBS|FBG|blood\s+sugar)\s*[:\-]?\s*([\d]+\.?\d*)", re.IGNORECASE),
    ],
    "hba1c": [
        re.compile(r"(?:hba1c|hb\s*a1c|glycated\s+hemoglobin|a1c)\s*[:\-]?\s*([\d]+\.?\d*)\s*%?", re.IGNORECASE),
    ],
    "ldl": [
        re.compile(r"(?:ldl|ldl[- ]?cholesterol|low\s+density)\s*[:\-]?\s*([\d]+\.?\d*)\s*(?:mg/dL|mg/dl)?", re.IGNORECASE),
    ],
    "hdl": [
        re.compile(r"(?:hdl|hdl[- ]?cholesterol|high\s+density)\s*[:\-]?\s*([\d]+\.?\d*)\s*(?:mg/dL|mg/dl)?", re.IGNORECASE),
    ],
    "total_cholesterol": [
        re.compile(r"(?:total\s+cholesterol|cholesterol\s*,?\s*total)\s*[:\-]?\s*([\d]+\.?\d*)\s*(?:mg/dL|mg/dl)?", re.IGNORECASE),
        re.compile(r"(?:cholesterol)\s*[:\-]?\s*([\d]+\.?\d*)\s*(?:mg/dL|mg/dl)", re.IGNORECASE),
    ],
    "triglycerides": [
        re.compile(r"(?:triglycerides?|tg|trigs?)\s*[:\-]?\s*([\d]+\.?\d*)\s*(?:mg/dL|mg/dl)?", re.IGNORECASE),
    ],
    "vitamin_d": [
        re.compile(r"(?:vitamin\s*d|vit\.?\s*d|25-?hydroxy|25\(?OH\)?D)\s*[:\-]?\s*([\d]+\.?\d*)\s*(?:ng/mL|ng/ml)?", re.IGNORECASE),
    ],
    "vitamin_b12": [
        re.compile(r"(?:vitamin\s*b\s*12|vit\.?\s*b\s*12|cobalamin|b12)\s*[:\-]?\s*([\d]+\.?\d*)\s*(?:pg/mL|pg/ml|pmol/L)?", re.IGNORECASE),
    ],
}


def extract_biomarkers(raw_text: str) -> dict[str, Optional[float]]:
    """
    Scan raw OCR text and extract biomarker values using regex.
    Returns a dict: { biomarker_name: numeric_value_or_None }
    """
    results: dict[str, Optional[float]] = {}

    for biomarker, patterns in BIOMARKER_PATTERNS.items():
        value = None
        for pattern in patterns:
            match = pattern.search(raw_text)
            if match:
                try:
                    value = float(match.group(1))
                except (ValueError, IndexError):
                    value = None
                break  # use first successful match
        results[biomarker] = value

    return results


def get_extraction_summary(biomarkers: dict[str, Optional[float]]) -> dict:
    """
    Return a summary of extraction: which biomarkers were found vs. missing.
    """
    found = {k: v for k, v in biomarkers.items() if v is not None}
    missing = [k for k, v in biomarkers.items() if v is None]

    return {
        "found_count": len(found),
        "missing_count": len(missing),
        "found": found,
        "missing": missing,
    }
