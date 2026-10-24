from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly the feature engineering used during model training."""

    data = df.copy()

    data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25
    data["EMPLOYED_YEARS"] = -data["DAYS_EMPLOYED"] / 365.25

    def safe_divide(a, b):
        return a / b.replace(0, np.nan)

    data["CREDIT_TO_INCOME"] = safe_divide(
        data["AMT_CREDIT"], data["AMT_INCOME_TOTAL"]
    )

    data["ANNUITY_TO_INCOME"] = safe_divide(
        data["AMT_ANNUITY"], data["AMT_INCOME_TOTAL"]
    )

    data["CREDIT_TO_ANNUITY"] = safe_divide(
        data["AMT_CREDIT"], data["AMT_ANNUITY"]
    )

    data["GOODS_TO_CREDIT"] = safe_divide(
        data["AMT_GOODS_PRICE"], data["AMT_CREDIT"]
    )

    data["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
        data["AMT_INCOME_TOTAL"], data["CNT_FAM_MEMBERS"]
    )

    data["INCOME_PER_CHILD"] = (
        data["AMT_INCOME_TOTAL"] /
        (data["CNT_CHILDREN"] + 1)
    )

    external_sources = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3"
    ]

    data["EXT_SOURCE_MEAN"] = data[
        external_sources
    ].mean(axis=1)

    data["EXT_SOURCE_MAX"] = data[
        external_sources
    ].max(axis=1)

    return data


class PredictionService:

    def __init__(self):
        self.model = joblib.load(
            MODEL_DIR / "baseline_xgboost.joblib"
        )

        self.preprocessor = joblib.load(
            MODEL_DIR / "baseline_preprocessor.joblib"
        )

        self.explainer = shap.TreeExplainer(
            self.model
        )
        self.feature_names = (
            self.preprocessor.get_feature_names_out()
        )

        # Recover the exact raw columns expected by the
        # preprocessing pipeline.
        self.expected_columns = []

        for _, _, columns in self.preprocessor.transformers_:
            self.expected_columns.extend(columns)

        print("Drifting Oracle model loaded.")
        print(
            "Expected raw features:",
            len(self.expected_columns)
        )

    def prepare_input(self, applicant: dict):

        # Remove API-only metadata.
        clean_data = {
            k: v for k, v in applicant.items()
            if k not in ["application_id", "name", "TARGET"]
        }

        # Create the COMPLETE model input schema.
        # Missing fields remain NaN and are handled by
        # the trained preprocessing pipeline.
        row = {
            column: np.nan
            for column in self.expected_columns
        }

        # Insert whatever applicant data was supplied.
        for column, value in clean_data.items():
            if column in row:
                row[column] = np.nan if value is None else value

        df = pd.DataFrame([row])

        # Known Home Credit sentinel.
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED"] = df[
                "DAYS_EMPLOYED"
            ].replace(365243, np.nan)

        # Same feature engineering as training.
        df = engineer_features(df)

        return df

    def predict(self, applicant: dict):

        df = self.prepare_input(applicant)

        X_processed = self.preprocessor.transform(df)

        probability = float(
            self.model.predict_proba(
                X_processed
            )[0][1]
        )

        return probability, X_processed

    def explain(self, applicant: dict, top_n: int = 5):

        df = self.prepare_input(applicant)
        X_processed = self.preprocessor.transform(df)
        shap_values = self.explainer.shap_values(
            X_processed
        )

        shap_array = np.asarray(shap_values)
        if shap_array.ndim != 2 or shap_array.shape != (
            1,
            len(self.feature_names),
        ):
            raise RuntimeError(
                "SHAP output does not align with the transformed feature schema."
            )
        values = shap_array[0]

        # Find strongest contributors
        importance = np.abs(values)
        top_indices = np.argsort(
            importance
        )[::-1][:top_n]

        factors = []
        for index in top_indices:
            factors.append({
                "feature": self.feature_names[index],
                "impact": (
                    "increases_risk"
                    if values[index] > 0
                    else "decreases_risk"
                ),
                "shap_value": round(
                    float(values[index]),
                    5
                )
            })

        return factors


prediction_service = PredictionService()