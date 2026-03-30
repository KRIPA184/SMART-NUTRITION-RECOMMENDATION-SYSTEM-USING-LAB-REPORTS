"""
Validation Service – Phase 1
Validates extracted biomarker values against expected units and plausible ranges.
Normalizes data for downstream ML models.
"""

from typing import Optional
from utils.config import BIOMARKER_RANGES


def validate_biomarkers(biomarkers: dict[str, Optional[float]]) -> dict:
    """
    Validate each biomarker value:
      - Check if within plausible physiological range
      - Flag out-of-range or critical values
      - Return status per biomarker (normal, low, high, critical, missing)
    
    Returns:
        {
            "validated": { biomarker: { "value": ..., "status": ..., "unit": ... } },
            "critical_flags": [ ... ],
            "warnings": [ ... ]
        }
    """
    validated: dict = {}
    critical_flags: list[str] = []
    warnings: list[str] = []

    for name, ref in BIOMARKER_RANGES.items():
        value = biomarkers.get(name)

        if value is None:
            validated[name] = {
                "value": None,
                "status": "missing",
                "unit": ref["unit"],
                "message": f"{name} not found in report",
            }
            warnings.append(f"{name} was not detected in the uploaded report.")
            continue

        entry: dict = {
            "value": value,
            "unit": ref["unit"],
        }

        # ── Critical value detection ──────────────────────────────
        if ref["critical_low"] is not None and value <= ref["critical_low"]:
            entry["status"] = "critical_low"
            entry["message"] = f"{name} is critically low ({value} {ref['unit']})"
            critical_flags.append(entry["message"])
        elif ref["critical_high"] is not None and value >= ref["critical_high"]:
            entry["status"] = "critical_high"
            entry["message"] = f"{name} is critically high ({value} {ref['unit']})"
            critical_flags.append(entry["message"])
        # ── Normal range check ────────────────────────────────────
        elif value < ref["low"]:
            entry["status"] = "low"
            entry["message"] = f"{name} is below normal range"
        elif value > ref["high"]:
            entry["status"] = "high"
            entry["message"] = f"{name} is above normal range"
        else:
            entry["status"] = "normal"
            entry["message"] = f"{name} is within normal range"

        validated[name] = entry

    return {
        "validated": validated,
        "critical_flags": critical_flags,
        "warnings": warnings,
    }


def normalize_for_model(biomarkers: dict[str, Optional[float]]) -> dict[str, float]:
    """
    Normalize biomarker values to 0-1 scale using reference ranges.
    Missing values are imputed with 0.5 (midpoint / normal).
    Used as input for ML models.
    """
    normalized: dict[str, float] = {}

    for name, ref in BIOMARKER_RANGES.items():
        value = biomarkers.get(name)
        low = ref["low"]
        high = ref["high"]
        span = high - low if high != low else 1.0

        if value is None:
            normalized[name] = 0.5  # impute as midpoint
        else:
            # Clamp then normalize
            clamped = max(low * 0.3, min(value, high * 2.5))
            normalized[name] = (clamped - low) / span

    return normalized
