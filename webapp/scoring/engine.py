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

        # Feature medians (default values for features not on the form)
        medians_path = model_dir / "feature_medians.json"
        if medians_path.exists():
            with open(medians_path) as f:
                self.feature_medians: dict = json.load(f)
        else:
            self.feature_medians = {}

        # Isotonic calibrator (optional — created in notebook section 14)
        calibrator_path = model_dir / "isotonic_calibrator.pkl"
        self.calibrator = (
            joblib.load(calibrator_path) if calibrator_path.exists() else None
        )

        # SHAP explainer (may fail if pickle was created with incompatible version)
        explainer_path = model_dir / "shap_explainer.pkl"
        try:
            self.shap_explainer = (
                joblib.load(explainer_path) if explainer_path.exists() else None
            )
        except Exception:
            import shap
            self.shap_explainer = shap.TreeExplainer(self.model)

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
        """Create a single-row dict with all 198 features."""
        # Start with median defaults for ALL features
        row: dict[str, Any] = {}
        for feat in self.feature_names:
            row[feat] = self.feature_medians.get(feat, np.nan)

        # Fill user-provided values (overrides medians)
        for key, value in user_input.items():
            if key in row:
                row[key] = value

        # Map form fields to model fields
        if "AGE_YEARS" in user_input:
            row["DAYS_BIRTH"] = user_input["AGE_YEARS"] * 365.25
        if "EMPLOYMENT_YEARS" in user_input:
            row["DAYS_EMPLOYED"] = user_input["EMPLOYMENT_YEARS"] * 365.25
            row["YEARS_EMPLOYED"] = user_input["EMPLOYMENT_YEARS"]
        if "ID_PUBLISH_YEARS" in user_input:
            row["DAYS_ID_PUBLISH"] = user_input["ID_PUBLISH_YEARS"] * 365.25
        if "REGISTRATION_YEARS" in user_input:
            row["DAYS_REGISTRATION"] = user_input["REGISTRATION_YEARS"] * 365.25
        if "PHONE_CHANGE_DAYS" in user_input:
            row["DAYS_LAST_PHONE_CHANGE"] = user_input["PHONE_CHANGE_DAYS"]

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
            elif "__MISSING__" in le.classes_:
                row[col] = le.transform(["__MISSING__"])[0]
            else:
                row[col] = np.nan

        # --- Engineered features ---
        self._engineer_financial(row)
        self._engineer_ext_source(row)
        self._engineer_social(row)
        self._engineer_age_group(row)
        self._engineer_contact(row)
        self._engineer_cross_features(row)

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
            row["CREDIT_GOODS_RATIO"] = amt_credit / amt_goods
        if amt_income and amt_credit:
            row["INCOME_CREDIT_PERC"] = amt_income / amt_credit
        if emp_years is not None and age_years and age_years > 0:
            row["EMPLOYED_TO_AGE_RATIO"] = emp_years / age_years

    @staticmethod
    def _engineer_ext_source(row: dict) -> None:
        e1 = row.get("EXT_SOURCE_1")
        e2 = row.get("EXT_SOURCE_2")
        e3 = row.get("EXT_SOURCE_3")
        sources = []
        for v in (e1, e2, e3):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                sources.append(float(v))
        if sources:
            row["EXT_SOURCES_MEAN"] = float(np.mean(sources))
            row["EXT_SOURCES_STD"] = float(np.std(sources)) if len(sources) > 1 else 0.0
            row["EXT_SOURCES_PROD"] = (
                float(np.prod(sources)) if len(sources) == 3 else np.nan
            )
        # Pairwise products
        if e1 is not None and e2 is not None:
            row["EXT_SOURCE_1x2"] = float(e1) * float(e2)
        if e2 is not None and e3 is not None:
            row["EXT_SOURCE_2x3"] = float(e2) * float(e3)
        if e1 is not None and e3 is not None:
            row["EXT_SOURCE_1x3"] = float(e1) * float(e3)

    @staticmethod
    def _engineer_social(row: dict) -> None:
        d30 = row.get("DEF_30_CNT_SOCIAL_CIRCLE", 0) or 0
        d60 = row.get("DEF_60_CNT_SOCIAL_CIRCLE", 0) or 0
        row["SOCIAL_CIRCLE_DEFAULT"] = max(d30, d60)

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

    @staticmethod
    def _engineer_cross_features(row: dict) -> None:
        """Cross-features between application data and aggregated bureau/prev/install/POS/CC."""
        amt_income = row.get("AMT_INCOME_TOTAL")
        amt_credit = row.get("AMT_CREDIT")
        amt_annuity = row.get("AMT_ANNUITY")

        # Bureau cross-features
        bureau_debt = row.get("BUREAU_AMT_DEBT_SUM")
        bureau_credit = row.get("BUREAU_AMT_CREDIT_SUM")
        if bureau_debt is not None and amt_income and amt_income > 0:
            row["BUREAU_DEBT_INCOME_RATIO"] = bureau_debt / amt_income
        if bureau_credit is not None and amt_credit and amt_credit > 0:
            row["BUREAU_CREDIT_VS_CURRENT"] = bureau_credit / amt_credit

        # Previous application cross-features
        prev_annuity = row.get("PREV_AVG_ANNUITY")
        if prev_annuity is not None and amt_annuity and amt_annuity > 0:
            row["PREV_ANNUITY_VS_CURRENT"] = prev_annuity / amt_annuity

        # Total loan count (bureau + POS + CC)
        bureau_loans = row.get("BUREAU_LOAN_COUNT", 0) or 0
        pos_contracts = row.get("POS_CONTRACT_COUNT", 0) or 0
        cc_cards = row.get("CC_CARD_COUNT", 0) or 0
        row["TOTAL_LOAN_COUNT"] = bureau_loans + pos_contracts + cc_cards

        # Good payment score (composite)
        bureau_closed = row.get("BUREAU_CLOSED_RATIO", 0.5) or 0.5
        install_late = row.get("INSTALL_LATE_RATIO", 0.5) or 0.5
        pos_completed = row.get("POS_COMPLETED_RATIO", 0.5) or 0.5
        cc_util = row.get("CC_UTILIZATION_MEAN", 0.5) or 0.5
        cc_util = max(0, min(1, cc_util))
        row["GOOD_PAYMENT_SCORE"] = (
            bureau_closed * 0.3
            + (1 - install_late) * 0.3
            + pos_completed * 0.2
            + (1 - cc_util) * 0.2
        )

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
        # Aggregated features filled by medians (not directly user-controlled)
        "BUREAU_AMT_CREDIT_SUM", "BUREAU_AMT_CREDIT_MEAN", "BUREAU_AMT_DEBT_SUM",
        "BUREAU_AMT_OVERDUE_SUM", "BUREAU_CREDIT_DURATION_MEAN", "BUREAU_DPD_TOTAL",
        "BUREAU_MONTHS_HISTORY", "BUREAU_DEBT_RATIO", "BUREAU_CLOSED_COUNT",
        "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT", "PREV_AVG_APPLICATION",
        "PREV_AVG_CREDIT", "PREV_MAX_CREDIT", "PREV_AVG_ANNUITY",
        "PREV_AVG_DOWN_PAYMENT", "PREV_DAYS_LAST_APP", "PREV_DAYS_FIRST_APP",
        "PREV_CREDIT_VS_APPLICATION",
        "INSTALL_COUNT", "INSTALL_DAYS_DIFF_MEAN", "INSTALL_DAYS_DIFF_MAX",
        "INSTALL_LATE_COUNT", "INSTALL_PAYMENT_RATIO_MEAN",
        "INSTALL_PAYMENT_RATIO_MIN", "INSTALL_AMT_PAYMENT_SUM",
        "INSTALL_AMT_INSTALMENT_SUM", "INSTALL_OVERALL_PAYMENT_RATIO",
        "POS_MONTHS_COUNT", "POS_DPD_MEAN", "POS_DPD_DEF_MAX",
        "POS_COMPLETED_COUNT", "POS_ACTIVE_COUNT", "POS_COMPLETED_RATIO",
        "CC_BALANCE_MAX", "CC_LIMIT_MEAN", "CC_UTILIZATION_MAX",
        "CC_DRAWINGS_COUNT", "CC_PAYMENT_TOTAL_MEAN", "CC_MIN_INSTALLMENT_MEAN",
        "CC_DPD_MAX", "CC_DPD_MEAN", "CC_MONTHS_COUNT", "CC_PAYMENT_VS_BALANCE",
        # Cross/derived features
        "BUREAU_DEBT_INCOME_RATIO", "BUREAU_CREDIT_VS_CURRENT",
        "PREV_ANNUITY_VS_CURRENT", "TOTAL_LOAN_COUNT", "GOOD_PAYMENT_SCORE",
        "EXT_SOURCES_STD", "EXT_SOURCES_PROD", "EXT_SOURCE_1x2",
        "EXT_SOURCE_2x3", "EXT_SOURCE_1x3", "INCOME_CREDIT_PERC",
        "SOCIAL_CIRCLE_DEFAULT",
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

