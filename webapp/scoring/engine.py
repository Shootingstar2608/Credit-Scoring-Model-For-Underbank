"""
Credit Scoring Inference Engine.
Loads model artifacts and performs prediction + feature engineering.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class CreditScoringEngine:
    """Stateful engine — loads model once, predicts many times."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "lgbm_credit_scoring.pkl")
        with open(model_dir / "feature_names.json") as f:
            self.feature_names: list[str] = json.load(f)
        self.label_encoders: dict = joblib.load(model_dir / "label_encoders.pkl")

        # Isotonic calibrator (optional — created in notebook section 14)
        calibrator_path = model_dir / "isotonic_calibrator.pkl"
        self.calibrator = (
            joblib.load(calibrator_path) if calibrator_path.exists() else None
        )

        # SHAP explainer
        explainer_path = model_dir / "shap_explainer.pkl"
        self.shap_explainer = (
            joblib.load(explainer_path) if explainer_path.exists() else None
        )

    # ------------------------------------------------------------------
    # FICO Scoring Parameters (log-odds based)
    # ------------------------------------------------------------------
    PDO = 40  # Points to Double the Odds (smaller = more aggressive)
    BASE_SCORE = 600  # Score at 50/50 odds (p=0.5)
    FACTOR = PDO / np.log(2)  # ≈57.71
    NO_HISTORY_PENALTY = 20  # Flat penalty: no credit bureau history
    EXT_DEFAULT_PENALTY = 15  # Per-source penalty for neutral/default values
    EXT_PERFECT_CAP = 750  # Max FICO when ≥2 sources are suspiciously perfect
    EXT_HIGH_THRESHOLD = 0.75  # Above this, dampen EXT_SOURCE bonus
    EXT_HIGH_MAX_PENALTY = 30  # Max dampening for very high EXT_SOURCE mean

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, user_input: dict[str, Any]) -> dict[str, Any]:
        """
        Full pipeline: raw user input → engineered features → prediction.

        Returns dict with keys:
            probability, credit_score, fico_score, risk_key, tier,
            shap_top_features, credit_limit_suggestion
        """
        row = self._build_feature_row(user_input)
        df_input = pd.DataFrame([row])[self.feature_names]
        for col in df_input.columns:
            df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

        # Raw probability (used for FICO — preserves spread & ranking)
        raw_proba = float(self.model.predict_proba(df_input)[:, 1][0])

        # Calibrated probability (for display — reflects true default rate)
        if self.calibrator is not None:
            cal_proba = float(self.calibrator.predict(np.array([raw_proba]))[0])
        else:
            cal_proba = raw_proba

        # FICO score uses RAW probability (wider distribution, better FICO spread)
        fico_score = self._probability_to_fico(raw_proba, user_input)
        risk_key = self._risk_key(fico_score)

        # SHAP explanations
        shap_top = self._shap_explain(df_input)

        # Credit limit suggestion (based on FICO)
        income = user_input.get("AMT_INCOME_TOTAL", 0) or 0
        credit_limit = self._credit_limit(fico_score, income)

        return {
            "probability": round(cal_proba, 4),
            "raw_probability": round(raw_proba, 4),
            "credit_score": fico_score,  # FICO 300-850
            "fico_score": fico_score,  # alias
            "risk_key": risk_key,
            "shap_top_features": shap_top,
            "credit_limit": credit_limit,
            "feature_vector": df_input,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_feature_row(self, user_input: dict) -> dict:
        """Create a single-row dict with all 48 features."""
        row: dict[str, Any] = {feat: np.nan for feat in self.feature_names}

        # Fill user-provided values
        for key, value in user_input.items():
            if key in row:
                row[key] = value

        # Encode categoricals
        for col, le in self.label_encoders.items():
            if col not in row:
                continue
            val = row[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            val_str = str(val)
            if val_str in le.classes_:
                row[col] = le.transform([val_str])[0]
            elif "MISSING" in le.classes_:
                row[col] = le.transform(["MISSING"])[0]
            else:
                row[col] = np.nan

        # --- Engineered features ---
        self._engineer_financial(row)
        self._engineer_ext_source(row)
        self._engineer_social(row)
        self._engineer_age_group(row)
        self._engineer_contact(row)

        return row

    @staticmethod
    def _engineer_financial(row: dict) -> None:
        amt_credit = row.get("AMT_CREDIT")
        amt_income = row.get("AMT_INCOME_TOTAL")
        amt_annuity = row.get("AMT_ANNUITY")
        amt_goods = row.get("AMT_GOODS_PRICE")
        emp_years = row.get("EMPLOYMENT_YEARS")
        age_years = row.get("AGE_YEARS")
        fam = row.get("CNT_FAM_MEMBERS")

        if amt_credit and amt_income:
            row["CREDIT_INCOME_RATIO"] = amt_credit / amt_income
        if amt_annuity and amt_income:
            row["ANNUITY_INCOME_RATIO"] = amt_annuity / amt_income
        if amt_credit and amt_annuity:
            row["CREDIT_TERM_MONTHS"] = amt_credit / amt_annuity
            row["PAYMENT_RATE"] = amt_annuity / amt_credit
        if amt_income and fam:
            row["INCOME_PER_PERSON"] = amt_income / fam
        if amt_goods and amt_credit:
            row["GOODS_CREDIT_RATIO"] = amt_goods / amt_credit
        if emp_years is not None and age_years and age_years > 0:
            row["EMPLOYED_TO_AGE_RATIO"] = emp_years / age_years

    @staticmethod
    def _engineer_ext_source(row: dict) -> None:
        sources = []
        for i in (1, 2, 3):
            v = row.get(f"EXT_SOURCE_{i}")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                sources.append(v)
        if sources:
            row["EXT_SOURCE_MEAN"] = float(np.mean(sources))
            row["EXT_SOURCE_PROD"] = (
                float(np.prod(sources)) if len(sources) == 3 else np.nan
            )
            row["EXT_SOURCE_MIN"] = float(np.min(sources))
            row["EXT_SOURCE_MAX"] = float(np.max(sources))

    @staticmethod
    def _engineer_social(row: dict) -> None:
        d30 = row.get("DEF_30_CNT_SOCIAL_CIRCLE", 0) or 0
        d60 = row.get("DEF_60_CNT_SOCIAL_CIRCLE", 0) or 0
        row["SOCIAL_DEF_TOTAL"] = d30 + d60

    @staticmethod
    def _engineer_age_group(row: dict) -> None:
        age = row.get("AGE_YEARS")
        if age is None:
            return
        if age < 27:
            row["AGE_GROUP"] = 0
        elif age < 35:
            row["AGE_GROUP"] = 1
        elif age < 45:
            row["AGE_GROUP"] = 2
        elif age < 55:
            row["AGE_GROUP"] = 3
        elif age < 65:
            row["AGE_GROUP"] = 4
        else:
            row["AGE_GROUP"] = 5

    @staticmethod
    def _engineer_contact(row: dict) -> None:
        flags = ["FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL"]
        row["CONTACT_COUNT"] = sum(1 for f in flags if row.get(f, 0) == 1)

    # ------------------------------------------------------------------
    def _probability_to_fico(self, proba: float, user_input: dict) -> int:
        """
        Convert default probability → FICO 300-850 using log-odds transform.

        Formula:  FICO = offset − factor × ln(odds)
        where odds = p / (1−p), factor = PDO / ln(2).

        Then apply:
          1) No-history penalty (flat −20)
          2) EXT_SOURCE reliability penalties
          3) EXT_SOURCE high-value dampening
        """
        eps = 1e-7
        p = np.clip(proba, eps, 1 - eps)
        odds = p / (1 - p)
        fico = self.BASE_SCORE - self.FACTOR * np.log(odds)

        # 1. No credit-history penalty
        fico -= self.NO_HISTORY_PENALTY

        # 2. EXT_SOURCE reliability: penalize neutral/default values
        ext_vals = []
        for i in (1, 2, 3):
            v = user_input.get(f"EXT_SOURCE_{i}", 0.5)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ext_vals.append(float(v))
            else:
                ext_vals.append(0.5)

        defaults = sum(1 for v in ext_vals if abs(v - 0.5) < 0.02)
        fico -= defaults * self.EXT_DEFAULT_PENALTY

        # 3. EXT_SOURCE high-value dampening (anti-gaming)
        perfect_count = sum(1 for v in ext_vals if v > 0.85)
        if perfect_count >= 2:
            fico = min(fico, self.EXT_PERFECT_CAP)

        ext_mean = float(np.mean(ext_vals))
        if ext_mean > self.EXT_HIGH_THRESHOLD:
            excess = (ext_mean - self.EXT_HIGH_THRESHOLD) / (
                1.0 - self.EXT_HIGH_THRESHOLD
            )
            fico -= int(excess * self.EXT_HIGH_MAX_PENALTY)

        return int(np.clip(round(fico), 300, 850))

    @staticmethod
    def _risk_key(fico_score: int) -> str:
        """Map FICO score to risk level key."""
        if fico_score >= 800:
            return "VERY_LOW"
        if fico_score >= 740:
            return "LOW"
        if fico_score >= 670:
            return "MEDIUM"
        if fico_score >= 580:
            return "HIGH"
        return "VERY_HIGH"

    @staticmethod
    def _credit_limit(fico_score: int, income: float) -> int:
        """Credit limit based on FICO tier, capped by income multiples."""
        from config import (
            BASE_CREDIT_LIMIT,
            INCOME_MULTIPLIER_MAX,
            MAX_CREDIT_LIMIT,
            MIN_CREDIT_LIMIT,
        )

        if fico_score < 580:
            return 0  # Too risky — reject tier

        # Higher FICO → higher multiplier
        if fico_score >= 800:
            tier_mult = 1.5
        elif fico_score >= 740:
            tier_mult = 1.2
        elif fico_score >= 670:
            tier_mult = 1.0
        else:  # 580-669 Fair
            tier_mult = 0.6

        income_factor = (
            min(income / 100_000, INCOME_MULTIPLIER_MAX) if income > 0 else 1
        )
        raw = BASE_CREDIT_LIMIT * tier_mult * income_factor
        return int(np.clip(raw, MIN_CREDIT_LIMIT, MAX_CREDIT_LIMIT))

    # ------------------------------------------------------------------
    # Features that are in the model but NOT on the input form
    # and NOT derivable from user input. These are NaN by default
    # and should be excluded from SHAP explanations.
    # ------------------------------------------------------------------
    HIDDEN_FEATURES = {
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "REGION_POPULATION_RELATIVE",
    }

    # ------------------------------------------------------------------
    def _shap_explain(self, df_input: pd.DataFrame) -> list[dict]:
        """Return top-10 SHAP contributors for this prediction."""
        if self.shap_explainer is None:
            return []
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                sv = self.shap_explainer.shap_values(df_input)
            if isinstance(sv, list):
                sv = sv[1]  # class 1 for binary
            if sv.ndim == 2:
                vals = sv[0]
            else:
                vals = sv
            features = self.feature_names
            pairs = sorted(zip(features, vals), key=lambda x: abs(x[1]), reverse=True)
            # Filter out hidden features (not controllable by user)
            pairs = [(f, v) for f, v in pairs if f not in self.HIDDEN_FEATURES]
            return [
                {
                    "feature": f,
                    "shap_value": round(float(v), 4),
                    "direction": "risk" if v > 0 else "safe",
                }
                for f, v in pairs[:10]
            ]
        except Exception:
            return []
