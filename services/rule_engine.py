"""
Rule Engine – Phase 3
Clinical rule-based safety filter, medication–nutrient interaction placeholder,
critical value detection, and multi-criteria decision scoring.
Generates the final structured health summary.
"""

from typing import Optional
from utils.config import BIOMARKER_RANGES, MEDICAL_DISCLAIMER


# ─── Medication–Nutrient Interaction Placeholders ─────────────────
MEDICATION_INTERACTIONS = {
    "warfarin":    {"avoid": ["Vitamin K-rich foods (spinach, kale, broccoli)"],
                    "note": "Vitamin K affects anticoagulant efficacy."},
    "metformin":   {"avoid": ["Excessive alcohol"],
                    "note": "May impair B12 absorption; monitor B12 levels."},
    "statins":     {"avoid": ["Grapefruit"],
                    "note": "Grapefruit can increase statin blood levels."},
    "lisinopril":  {"avoid": ["High-potassium foods in excess"],
                    "note": "ACE inhibitors may raise potassium levels."},
}


def compute_risk_score(validated: dict) -> dict:
    """
    Multi-criteria decision scoring based on validated biomarkers.
    Returns overall risk score (0-100) and category.
    """
    score = 0
    max_score = 0
    details: list[dict] = []

    # Weights for each biomarker (importance in health assessment)
    weights = {
        "hemoglobin": 10, "glucose": 15, "hba1c": 15,
        "ldl": 10, "hdl": 10, "total_cholesterol": 8,
        "triglycerides": 8, "vitamin_d": 8, "vitamin_b12": 6,
    }

    for name, entry in validated.items():
        w = weights.get(name, 5)
        max_score += w

        status = entry.get("status", "missing")
        if status == "normal":
            contrib = 0
        elif status == "missing":
            contrib = w * 0.3  # partial penalty for unknown
        elif status in ("low", "high"):
            contrib = w * 0.6
        elif status in ("critical_low", "critical_high"):
            contrib = w * 1.0
        else:
            contrib = 0

        score += contrib
        details.append({"biomarker": name, "status": status, "weight": w, "score_contribution": round(contrib, 2)})

    risk_pct = round((score / max_score) * 100, 1) if max_score else 0

    if risk_pct <= 25:
        category = "Low Risk"
    elif risk_pct <= 55:
        category = "Moderate Risk"
    else:
        category = "High Risk"

    return {
        "risk_score": risk_pct,
        "risk_category": category,
        "scoring_details": details,
    }


def check_medication_interactions(medications: list[str]) -> list[dict]:
    """
    Placeholder logic: check if any reported medications have known
    nutrient interactions. Returns advisory notes.
    """
    interactions = []
    for med in medications:
        key = med.strip().lower()
        if key in MEDICATION_INTERACTIONS:
            interactions.append({
                "medication": med,
                **MEDICATION_INTERACTIONS[key],
            })
    return interactions


# ─── Supplement Recommendation Engine ─────────────────────────────
SUPPLEMENT_MAP = {
    "hemoglobin": {
        "low": [
            {"name": "Ferrous Sulfate (Iron)", "dosage": "325 mg once daily", "form": "Tablet",
             "reason": "Iron supplementation helps restore hemoglobin levels and treat iron-deficiency anemia.",
             "precaution": "Take on an empty stomach with vitamin C for better absorption. May cause constipation."},
            {"name": "Folic Acid", "dosage": "400–800 mcg daily", "form": "Tablet",
             "reason": "Supports red blood cell production and works synergistically with iron.",
             "precaution": "Generally well-tolerated. Consult doctor if taking anticonvulsants."},
        ],
        "critical_low": [
            {"name": "Ferrous Sulfate (Iron)", "dosage": "325 mg twice daily", "form": "Tablet",
             "reason": "Higher-dose iron for critically low hemoglobin. Medical supervision recommended.",
             "precaution": "Seek immediate medical attention. IV iron may be required."},
            {"name": "Vitamin B12 (Methylcobalamin)", "dosage": "1000 mcg daily", "form": "Sublingual tablet",
             "reason": "B12 deficiency can compound anemia; supplementation supports RBC formation.",
             "precaution": "Safe at this dose. Sublingual form bypasses absorption issues."},
        ],
    },
    "glucose": {
        "high": [
            {"name": "Chromium Picolinate", "dosage": "200–400 mcg daily", "form": "Tablet",
             "reason": "May improve insulin sensitivity and support healthy blood sugar levels.",
             "precaution": "Consult doctor if on diabetes medication; may enhance hypoglycemic effects."},
            {"name": "Alpha-Lipoic Acid", "dosage": "300–600 mg daily", "form": "Capsule",
             "reason": "An antioxidant that may help improve glucose uptake and reduce oxidative stress.",
             "precaution": "May lower blood sugar; monitor glucose levels closely."},
        ],
    },
    "hba1c": {
        "high": [
            {"name": "Berberine", "dosage": "500 mg two to three times daily", "form": "Capsule",
             "reason": "Shown in studies to help lower HbA1c by improving insulin sensitivity.",
             "precaution": "May interact with diabetes medications. Take with meals to avoid GI discomfort."},
            {"name": "Magnesium Glycinate", "dosage": "200–400 mg daily", "form": "Tablet",
             "reason": "Magnesium deficiency is common in diabetics; supplementation may improve glycemic control.",
             "precaution": "High doses may cause loose stools. Reduce dose if GI issues occur."},
        ],
    },
    "ldl": {
        "high": [
            {"name": "Omega-3 Fish Oil", "dosage": "1000–2000 mg EPA+DHA daily", "form": "Softgel",
             "reason": "Omega-3 fatty acids help reduce triglycerides and may modestly lower LDL.",
             "precaution": "Choose a purified/pharmaceutical-grade product. May increase bleeding risk if on anticoagulants."},
            {"name": "Plant Sterols / Stanols", "dosage": "2 g daily", "form": "Tablet or fortified food",
             "reason": "Clinically shown to reduce LDL cholesterol absorption by 5–15%.",
             "precaution": "Take with meals. May slightly reduce absorption of fat-soluble vitamins."},
        ],
    },
    "hdl": {
        "low": [
            {"name": "Niacin (Vitamin B3)", "dosage": "500–1000 mg daily", "form": "Extended-release tablet",
             "reason": "Niacin is one of the most effective supplements for raising HDL cholesterol.",
             "precaution": "Start with low dose to minimize flushing. Monitor liver function with prolonged use."},
            {"name": "Omega-3 Fish Oil", "dosage": "1000–2000 mg EPA+DHA daily", "form": "Softgel",
             "reason": "Supports cardiovascular health and may modestly raise HDL.",
             "precaution": "May cause fishy aftertaste. Take with meals."},
        ],
    },
    "total_cholesterol": {
        "high": [
            {"name": "Red Yeast Rice", "dosage": "1200 mg daily", "form": "Capsule",
             "reason": "Contains natural monacolins that may help lower total cholesterol.",
             "precaution": "Do not combine with statin medications. Monitor liver enzymes."},
        ],
    },
    "triglycerides": {
        "high": [
            {"name": "Omega-3 Fish Oil (High-Dose)", "dosage": "2000–4000 mg EPA+DHA daily", "form": "Softgel",
             "reason": "High-dose omega-3 is clinically proven to significantly reduce elevated triglycerides.",
             "precaution": "Use pharmaceutical-grade product at this dose. Consult doctor if on blood thinners."},
        ],
    },
    "vitamin_d": {
        "low": [
            {"name": "Vitamin D3 (Cholecalciferol)", "dosage": "2000–4000 IU daily", "form": "Softgel or tablet",
             "reason": "D3 is the most effective form for restoring vitamin D levels.",
             "precaution": "Take with a fat-containing meal for better absorption. Recheck levels after 8–12 weeks."},
        ],
        "critical_low": [
            {"name": "Vitamin D3 (Cholecalciferol)", "dosage": "5000–10000 IU daily for 8 weeks, then maintenance", "form": "Softgel",
             "reason": "High-dose loading phase needed for critically low vitamin D, followed by maintenance dosing.",
             "precaution": "Requires medical supervision. Monitor calcium and vitamin D levels regularly."},
            {"name": "Calcium Citrate", "dosage": "500–600 mg daily", "form": "Tablet",
             "reason": "Vitamin D and calcium work together; calcium absorption is impaired when D is critically low.",
             "precaution": "Split doses for better absorption. Avoid taking with iron supplements."},
        ],
    },
    "vitamin_b12": {
        "low": [
            {"name": "Vitamin B12 (Methylcobalamin)", "dosage": "1000 mcg daily", "form": "Sublingual tablet",
             "reason": "Methylcobalamin is the bioactive form of B12 and is well absorbed sublingually.",
             "precaution": "Very safe at this dose. Sublingual form is preferred for absorption issues."},
        ],
        "critical_low": [
            {"name": "Vitamin B12 (Methylcobalamin)", "dosage": "2000 mcg daily or weekly injections", "form": "Sublingual tablet / Injection",
             "reason": "Critically low B12 may require high-dose oral or intramuscular injection therapy.",
             "precaution": "Consult doctor immediately. Injections may be necessary if oral absorption is impaired."},
            {"name": "Folic Acid", "dosage": "400 mcg daily", "form": "Tablet",
             "reason": "Folate and B12 work together in red blood cell synthesis; co-supplementation is often needed.",
             "precaution": "Do not take high-dose folate without B12 as it can mask B12 deficiency."},
        ],
    },
}


def recommend_supplements(validation_result: dict) -> list[dict]:
    """
    Based on validated biomarker statuses, recommend appropriate
    nutrition supplements or tablets with dosage and precautions.
    """
    recommendations = []
    validated = validation_result.get("validated", {})

    for biomarker_name, entry in validated.items():
        status = entry.get("status", "normal")
        if status in ("normal", "missing"):
            continue

        mapping = SUPPLEMENT_MAP.get(biomarker_name, {})
        supps = mapping.get(status, [])

        for supp in supps:
            recommendations.append({
                "biomarker": biomarker_name,
                "biomarker_status": status,
                "biomarker_value": f"{entry.get('value')} {entry.get('unit', '')}",
                **supp,
            })

    return recommendations


def generate_health_summary(
    biomarkers: dict[str, Optional[float]],
    validation_result: dict,
    risk_result: dict,
    patient_info: dict,
    nutrient_targets: dict,
    shap_explanation: dict,
    medications: Optional[list[str]] = None,
) -> dict:
    """
    Build the final structured health summary (Phase 3 output).
    This JSON is what gets sent to the LLM — never raw OCR text.
    """
    med_interactions = check_medication_interactions(medications or [])

    # ── Assemble critical alerts ──────────────────────────────────
    critical_flags = validation_result.get("critical_flags", [])
    has_emergency = len(critical_flags) > 0

    # ── Abnormal biomarkers list ──────────────────────────────────
    abnormal = []
    for name, entry in validation_result.get("validated", {}).items():
        if entry["status"] not in ("normal", "missing"):
            abnormal.append({
                "biomarker": name,
                "value": entry["value"],
                "unit": entry["unit"],
                "status": entry["status"],
            })

    supplement_recs = recommend_supplements(validation_result)

    summary = {
        "patient_info": patient_info,
        "biomarkers": biomarkers,
        "validation": validation_result,
        "risk_assessment": risk_result,
        "abnormal_biomarkers": abnormal,
        "nutrient_targets": nutrient_targets,
        "shap_top_features": shap_explanation.get("top_features", []),
        "medication_interactions": med_interactions,
        "supplement_recommendations": supplement_recs,
        "critical_alerts": critical_flags,
        "has_emergency": has_emergency,
        "disclaimer": MEDICAL_DISCLAIMER,
    }

    return summary
