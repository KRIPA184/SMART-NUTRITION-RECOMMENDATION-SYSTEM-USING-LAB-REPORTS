"""
Health Classification Model – Phase 2
Uses a RandomForest classifier to categorize patients into
Low Risk / Moderate Risk / High Risk based on biomarker values.
Includes SHAP-based explainability.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# ─── Labels ───────────────────────────────────────────────────────
RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
FEATURE_NAMES = [
    "hemoglobin", "glucose", "hba1c", "ldl", "hdl",
    "total_cholesterol", "triglycerides", "vitamin_d", "vitamin_b12",
]


def _generate_synthetic_data(n_samples: int = 500, seed: int = 42) -> tuple:
    """
    Generate synthetic training data for the health classifier.
    In production, this would be replaced with real clinical data.
    """
    rng = np.random.RandomState(seed)

    X = np.column_stack([
        rng.uniform(8, 20, n_samples),      # hemoglobin (g/dL)
        rng.uniform(50, 350, n_samples),     # glucose (mg/dL)
        rng.uniform(4, 12, n_samples),       # hba1c (%)
        rng.uniform(50, 220, n_samples),     # ldl (mg/dL)
        rng.uniform(20, 90, n_samples),      # hdl (mg/dL)
        rng.uniform(100, 350, n_samples),    # total cholesterol (mg/dL)
        rng.uniform(50, 500, n_samples),     # triglycerides (mg/dL)
        rng.uniform(5, 100, n_samples),      # vitamin D (ng/mL)
        rng.uniform(80, 1200, n_samples),    # vitamin B12 (pg/mL)
    ])

    # Create labels based on simple clinical heuristics
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        risk = 0
        if X[i, 1] > 126:  risk += 1       # glucose high
        if X[i, 2] > 6.5:  risk += 1       # hba1c high
        if X[i, 3] > 160:  risk += 1       # ldl high
        if X[i, 4] < 40:   risk += 1       # hdl low
        if X[i, 6] > 200:  risk += 1       # triglycerides high
        if X[i, 0] < 10:   risk += 1       # hemoglobin low
        if X[i, 7] < 20:   risk += 1       # vitamin D low

        if risk >= 3:
            y[i] = 2  # High Risk
        elif risk >= 1:
            y[i] = 1  # Moderate Risk
        else:
            y[i] = 0  # Low Risk

    return X, y


class HealthClassifier:
    """RandomForest-based health risk classifier with SHAP explainability."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )
        self._is_trained = False

    def train(self):
        """Train on synthetic data (replace with real data in production)."""
        X, y = _generate_synthetic_data()
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, biomarkers: dict[str, Optional[float]]) -> dict:
        """
        Predict health risk category from biomarker values.
        Returns: { "category": str, "probabilities": dict, "risk_label": str }
        """
        if not self._is_trained:
            self.train()

        features = self._prepare_features(biomarkers)
        X = np.array(features).reshape(1, -1)

        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        return {
            "category_id": int(prediction),
            "risk_label": RISK_LABELS[int(prediction)],
            "probabilities": {
                RISK_LABELS[i]: round(float(p), 4) for i, p in enumerate(probabilities)
            },
        }

    def get_shap_explanation(self, biomarkers: dict[str, Optional[float]]) -> dict:
        """
        Compute feature importance using tree-based SHAP (TreeExplainer).
        Returns top contributing biomarkers.
        """
        if not self._is_trained:
            self.train()

        try:
            import shap
            features = self._prepare_features(biomarkers)
            X = np.array(features).reshape(1, -1)

            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            # For multi-class, shap_values is a list of arrays (one per class)
            # Use the predicted class's SHAP values
            prediction = self.model.predict(X)[0]

            sv_array = np.array(shap_values)
            if sv_array.ndim == 3:
                # Shape (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
                if sv_array.shape[0] == X.shape[0]:
                    # (n_samples, n_features, n_classes)
                    sv = np.abs(sv_array[0, :, prediction])
                else:
                    # (n_classes, n_samples, n_features)
                    sv = np.abs(sv_array[prediction, 0, :])
            elif isinstance(shap_values, list):
                sv = np.abs(shap_values[prediction][0])
            else:
                sv = np.abs(sv_array[0])
            
            # Ensure sv is a flat 1D array of floats
            sv = np.asarray(sv).flatten()

            # Rank features by importance
            feature_importance = sorted(
                zip(FEATURE_NAMES, sv.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )

            top_features = [
                {"feature": name, "importance": round(imp, 4)}
                for name, imp in feature_importance
            ]

            return {"top_features": top_features, "method": "SHAP TreeExplainer"}

        except ImportError:
            # Fallback to built-in feature importance if SHAP unavailable
            importances = self.model.feature_importances_
            feature_importance = sorted(
                zip(FEATURE_NAMES, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            top_features = [
                {"feature": name, "importance": round(imp, 4)}
                for name, imp in feature_importance
            ]
            return {"top_features": top_features, "method": "RandomForest feature_importances_"}

    def _prepare_features(self, biomarkers: dict[str, Optional[float]]) -> list[float]:
        """
        Convert biomarker dict to ordered feature list.
        Impute missing values with midpoint of normal range.
        """
        from utils.config import BIOMARKER_RANGES

        features = []
        for name in FEATURE_NAMES:
            value = biomarkers.get(name)
            if value is not None:
                features.append(float(value))
            else:
                ref = BIOMARKER_RANGES.get(name, {})
                midpoint = (ref.get("low", 0) + ref.get("high", 100)) / 2
                features.append(midpoint)
        return features


# ─── Module-level singleton ───────────────────────────────────────
health_classifier = HealthClassifier()
