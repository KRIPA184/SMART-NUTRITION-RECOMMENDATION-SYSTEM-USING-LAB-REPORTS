"""
Nutrient Target Prediction Model – Phase 2
Uses LightGBM (or simplified regression fallback) to predict
personalized daily nutrient targets based on biomarkers + patient info.
"""

import numpy as np
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

NUTRIENT_NAMES = [
    "calories", "protein", "carbohydrates", "fat", "fiber",
    "iron", "calcium", "vitamin_d_intake", "vitamin_b12_intake", "omega_3",
]

FEATURE_NAMES = [
    "hemoglobin", "glucose", "hba1c", "ldl", "hdl",
    "total_cholesterol", "triglycerides", "vitamin_d", "vitamin_b12",
    "age", "gender_numeric", "activity_numeric",
]


def _gender_to_numeric(gender: str) -> float:
    g = gender.strip().lower()
    if g in ("male", "m"):
        return 1.0
    elif g in ("female", "f"):
        return 0.0
    return 0.5


def _activity_to_numeric(level: str) -> float:
    mapping = {
        "sedentary": 0.2, "light": 0.4, "lightly active": 0.4,
        "moderate": 0.6, "moderately active": 0.6,
        "active": 0.8, "very active": 1.0,
    }
    return mapping.get(level.strip().lower(), 0.5)


def _generate_synthetic_data(n_samples: int = 600, seed: int = 99) -> tuple:
    """Generate synthetic (X, Y) for nutrient target regression."""
    rng = np.random.RandomState(seed)

    X = np.column_stack([
        rng.uniform(8, 20, n_samples),       # hemoglobin
        rng.uniform(50, 350, n_samples),      # glucose
        rng.uniform(4, 12, n_samples),        # hba1c
        rng.uniform(50, 220, n_samples),      # ldl
        rng.uniform(20, 90, n_samples),       # hdl
        rng.uniform(100, 350, n_samples),     # total cholesterol
        rng.uniform(50, 500, n_samples),      # triglycerides
        rng.uniform(5, 100, n_samples),       # vitamin D
        rng.uniform(80, 1200, n_samples),     # vitamin B12
        rng.uniform(18, 80, n_samples),       # age
        rng.choice([0, 0.5, 1.0], n_samples), # gender
        rng.uniform(0.2, 1.0, n_samples),     # activity level
    ])

    # Target nutrients – heuristic-based generation
    Y = np.column_stack([
        1600 + X[:, 11] * 600 + X[:, 9] * 5 + rng.normal(0, 50, n_samples),   # calories
        50 + X[:, 11] * 40 + X[:, 10] * 15 + rng.normal(0, 5, n_samples),      # protein
        220 + X[:, 11] * 80 - X[:, 2] * 5 + rng.normal(0, 10, n_samples),      # carbs
        50 + X[:, 11] * 20 - X[:, 3] * 0.05 + rng.normal(0, 3, n_samples),     # fat
        28 + X[:, 11] * 5 + rng.normal(0, 2, n_samples),                        # fiber
        10 + (1 - X[:, 10]) * 8 + rng.normal(0, 1, n_samples),                  # iron
        1000 + X[:, 9] * 2 + rng.normal(0, 50, n_samples),                      # calcium
        800 + (50 - np.clip(X[:, 7], 5, 50)) * 20 + rng.normal(0, 50, n_samples),# vit d intake
        3 + (500 - np.clip(X[:, 8], 80, 500)) * 0.01 + rng.normal(0, 0.3, n_samples),  # vit b12 intake
        1.5 + X[:, 11] * 1.0 + rng.normal(0, 0.2, n_samples),                  # omega 3
    ])

    return X, Y


class NutrientPredictor:
    """Predicts personalized daily nutrient targets."""

    def __init__(self):
        self.models = []
        self._is_trained = False
        self._use_lgbm = False

    def train(self):
        """Train one model per nutrient target. Prefer LightGBM, fallback to sklearn."""
        X, Y = _generate_synthetic_data()

        try:
            from lightgbm import LGBMRegressor
            self._use_lgbm = True
            for i in range(Y.shape[1]):
                m = LGBMRegressor(n_estimators=80, max_depth=5, random_state=42, verbose=-1)
                m.fit(X, Y[:, i])
                self.models.append(m)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self._use_lgbm = False
            for i in range(Y.shape[1]):
                m = GradientBoostingRegressor(n_estimators=80, max_depth=5, random_state=42)
                m.fit(X, Y[:, i])
                self.models.append(m)

        self._is_trained = True

    def predict(
        self,
        biomarkers: dict[str, Optional[float]],
        patient_info: dict,
    ) -> dict:
        """
        Predict daily nutrient targets.
        Returns: { nutrient_name: { "value": ..., "unit": ... } }
        """
        if not self._is_trained:
            self.train()

        features = self._prepare_features(biomarkers, patient_info)
        X = np.array(features).reshape(1, -1)

        units = {
            "calories": "kcal", "protein": "g", "carbohydrates": "g",
            "fat": "g", "fiber": "g", "iron": "mg", "calcium": "mg",
            "vitamin_d_intake": "IU", "vitamin_b12_intake": "mcg", "omega_3": "g",
        }

        targets = {}
        for i, name in enumerate(NUTRIENT_NAMES):
            pred = float(self.models[i].predict(X)[0])
            targets[name] = {
                "value": round(max(0, pred), 1),
                "unit": units.get(name, ""),
            }

        return targets

    def _prepare_features(
        self,
        biomarkers: dict[str, Optional[float]],
        patient_info: dict,
    ) -> list[float]:
        """Build feature vector from biomarkers + patient demographics."""
        from utils.config import BIOMARKER_RANGES

        bio_features = []
        bio_names = [
            "hemoglobin", "glucose", "hba1c", "ldl", "hdl",
            "total_cholesterol", "triglycerides", "vitamin_d", "vitamin_b12",
        ]
        for name in bio_names:
            value = biomarkers.get(name)
            if value is not None:
                bio_features.append(float(value))
            else:
                ref = BIOMARKER_RANGES.get(name, {})
                bio_features.append((ref.get("low", 0) + ref.get("high", 100)) / 2)

        age = float(patient_info.get("age", 35))
        gender_num = _gender_to_numeric(patient_info.get("gender", "other"))
        activity_num = _activity_to_numeric(patient_info.get("activity_level", "moderate"))

        return bio_features + [age, gender_num, activity_num]


# ─── Module-level singleton ───────────────────────────────────────
nutrient_predictor = NutrientPredictor()
