"""
Health Score Service
Computes category-wise health scores (0–100) from validated biomarker data.

Categories:
  • Metabolic Health   – glucose, hba1c
  • Heart Health       – ldl, hdl, total_cholesterol, triglycerides
  • Vitamin Balance    – vitamin_d, vitamin_b12
  • Inflammation Index – hemoglobin (proxy for chronic inflammation / anemia)

Each biomarker contributes a sub-score based on how close it is to the
optimal range. The overall Health Score is a weighted average.
"""

from utils.config import BIOMARKER_RANGES

# ─── Category Definitions ─────────────────────────────────────────
# { category_name: [ (biomarker, weight), ... ] }
HEALTH_CATEGORIES = {
    "Metabolic Health": [
        ("glucose", 1.0),
        ("hba1c", 1.0),
    ],
    "Heart Health": [
        ("ldl", 1.0),
        ("hdl", 1.0),
        ("total_cholesterol", 0.8),
        ("triglycerides", 0.8),
    ],
    "Vitamin Balance": [
        ("vitamin_d", 1.0),
        ("vitamin_b12", 1.0),
    ],
    "Inflammation Index": [
        ("hemoglobin", 1.0),
    ],
}

# Weights for overall score
CATEGORY_WEIGHTS = {
    "Metabolic Health": 30,
    "Heart Health": 30,
    "Vitamin Balance": 20,
    "Inflammation Index": 20,
}


def _biomarker_score(name: str, value) -> float:
    """
    Score a single biomarker on a 0–100 scale.
    100 = perfectly in the middle of normal range.
    Degrades linearly as it moves toward critical boundaries.
    """
    if value is None:
        return 50.0  # unknown → neutral

    ref = BIOMARKER_RANGES.get(name)
    if not ref:
        return 50.0

    low = ref["low"]
    high = ref["high"]
    crit_low = ref.get("critical_low")
    crit_high = ref.get("critical_high")

    val = float(value)

    # Perfect: inside normal range
    if low <= val <= high:
        # Score 80-100: best at midpoint
        mid = (low + high) / 2
        span = (high - low) / 2 if high != low else 1
        distance = abs(val - mid) / span  # 0 at mid, 1 at edge
        return round(100 - distance * 20, 1)  # 80–100

    # Below normal
    if val < low:
        floor = crit_low if crit_low is not None else low * 0.3
        if floor >= low:
            floor = low * 0.3
        rng = low - floor
        if rng <= 0:
            return 30.0
        dist = max(0, low - val)
        pct = min(dist / rng, 1.0)
        return round(80 - pct * 60, 1)  # 80 → 20

    # Above normal
    ceiling = crit_high if crit_high is not None else high * 2.5
    if ceiling <= high:
        ceiling = high * 2.5
    rng = ceiling - high
    if rng <= 0:
        return 30.0
    dist = max(0, val - high)
    pct = min(dist / rng, 1.0)
    return round(80 - pct * 60, 1)  # 80 → 20


def compute_health_scores(validated: dict) -> dict:
    """
    Compute per-category health scores and an overall Health Score.

    Args:
        validated: dict from validation_result["validated"]
                   { biomarker: { "value": ..., "status": ..., "unit": ... } }

    Returns:
        {
          "overall": { "score": 72.5, "label": "Good", "color": "success" },
          "categories": {
              "Metabolic Health": { "score": 65.0, "label": "Fair", ... },
              ...
          },
          "biomarker_scores": { "glucose": 85.0, ... }
        }
    """
    biomarker_scores: dict[str, float] = {}
    category_results: dict[str, dict] = {}

    for cat_name, members in HEALTH_CATEGORIES.items():
        total_weighted = 0.0
        total_weight = 0.0

        for bm_name, bm_weight in members:
            entry = validated.get(bm_name, {})
            val = entry.get("value")
            sc = _biomarker_score(bm_name, val)
            biomarker_scores[bm_name] = sc
            total_weighted += sc * bm_weight
            total_weight += bm_weight

        cat_score = round(total_weighted / total_weight, 1) if total_weight else 50.0
        category_results[cat_name] = {
            "score": cat_score,
            "label": _score_label(cat_score),
            "color": _score_color(cat_score),
        }

    # Overall weighted average
    overall_total = 0.0
    overall_weight = 0.0
    for cat_name, cat_w in CATEGORY_WEIGHTS.items():
        cat_sc = category_results.get(cat_name, {}).get("score", 50)
        overall_total += cat_sc * cat_w
        overall_weight += cat_w

    overall_score = round(overall_total / overall_weight, 1) if overall_weight else 50.0

    return {
        "overall": {
            "score": overall_score,
            "label": _score_label(overall_score),
            "color": _score_color(overall_score),
        },
        "categories": category_results,
        "biomarker_scores": biomarker_scores,
    }


def _score_label(score: float) -> str:
    if score >= 80:
        return "Excellent"
    elif score >= 65:
        return "Good"
    elif score >= 50:
        return "Fair"
    elif score >= 35:
        return "Needs Attention"
    else:
        return "Critical"


def _score_color(score: float) -> str:
    """Bootstrap color class."""
    if score >= 80:
        return "success"
    elif score >= 65:
        return "info"
    elif score >= 50:
        return "warning"
    else:
        return "danger"
