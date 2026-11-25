from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import pandas as pd
import numpy as np
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
V2_MODEL_DIR = PROJECT_ROOT / "fine_tuning" / "artifacts"

ENGINEERED_FEATURES = {
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "CREDIT_TO_INCOME",
    "ANNUITY_TO_INCOME",
    "CREDIT_TO_ANNUITY",
    "GOODS_TO_CREDIT",
    "INCOME_PER_FAMILY_MEMBER",
    "INCOME_PER_CHILD",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MAX",
}

MODEL_ARTIFACTS = {
    "v1": (
        MODEL_DIR / "baseline_xgboost.joblib",
        MODEL_DIR / "baseline_preprocessor.joblib",
        "baseline-v1",
    ),
    "v2": (
        V2_MODEL_DIR / "v2_xgboost_model.joblib",
        V2_MODEL_DIR / "v2_preprocessor.joblib",
        "v2",
    ),
}


@dataclass(frozen=True)
class ModelBundle:
    version: str
    label: str
    model: Any
    preprocessor: Any
    explainer: Any
    input_columns: tuple[str, ...]
    raw_feature_columns: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: frozenset[str]
    feature_names: tuple[str, ...]


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
        self._models: dict[str, ModelBundle] = {}
        self._model_lock = RLock()
        baseline = self._load_model("v1")
        self._models["v1"] = baseline

        # Preserve the V1 attributes used by existing callers.
        self.model = baseline.model
        self.preprocessor = baseline.preprocessor
        self.explainer = baseline.explainer
        self.feature_names = list(baseline.feature_names)
        self.expected_columns = list(baseline.input_columns)

        print("Drifting Oracle model loaded.")
        print(
            "Expected original features:",
            len(baseline.raw_feature_columns),
        )

    @staticmethod
    def normalize_version(model_version: str = "v1") -> str:
        aliases = {
            "v1": "v1",
            "baseline-v1": "v1",
            "v2": "v2",
        }
        try:
            return aliases[model_version]
        except KeyError as exc:
            raise ValueError(
                "model_version must be 'v1' or 'v2'."
            ) from exc

    def _load_model(self, model_version: str) -> ModelBundle:
        model_path, preprocessor_path, label = MODEL_ARTIFACTS[model_version]
        if not model_path.is_file() or not preprocessor_path.is_file():
            raise FileNotFoundError(
                f"Model artifacts are incomplete for {model_version}."
            )
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        input_columns = tuple(str(name) for name in preprocessor.feature_names_in_)
        feature_names = tuple(
            str(name) for name in preprocessor.get_feature_names_out()
        )
        if model.n_features_in_ != len(feature_names):
            raise RuntimeError(
                f"Model and preprocessor feature counts do not match for {model_version}."
            )
        raw_feature_columns = tuple(
            name for name in input_columns if name not in ENGINEERED_FEATURES
        )
        numeric_features = tuple(
            str(name)
            for transformer_name, _, columns in preprocessor.transformers_
            if transformer_name == "num"
            for name in columns
            if name not in ENGINEERED_FEATURES
        )
        categorical_features = frozenset(
            str(name)
            for transformer_name, _, columns in preprocessor.transformers_
            if transformer_name == "cat"
            for name in columns
            if name not in ENGINEERED_FEATURES
        )
        if set(raw_feature_columns) != set(numeric_features) | set(
            categorical_features
        ):
            raise RuntimeError(
                f"Preprocessor input schema is inconsistent for {model_version}."
            )
        return ModelBundle(
            version=model_version,
            label=label,
            model=model,
            preprocessor=preprocessor,
            explainer=shap.TreeExplainer(model),
            input_columns=input_columns,
            raw_feature_columns=raw_feature_columns,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            feature_names=feature_names,
        )

    def get_model(self, model_version: str = "v1") -> ModelBundle:
        version = self.normalize_version(model_version)
        bundle = self._models.get(version)
        if bundle is not None:
            return bundle
        with self._model_lock:
            bundle = self._models.get(version)
            if bundle is None:
                bundle = self._load_model(version)
                self._models[version] = bundle
        return bundle

    def model_label(self, model_version: str = "v1") -> str:
        return self.get_model(model_version).label

    def prepare_frame(
        self,
        frame: pd.DataFrame,
        model_version: str = "v1",
    ) -> pd.DataFrame:
        bundle = self.get_model(model_version)
        raw_features = frame.reindex(columns=bundle.raw_feature_columns).copy()
        raw_features = raw_features.where(pd.notna(raw_features), np.nan)
        if "DAYS_EMPLOYED" in raw_features:
            raw_features["DAYS_EMPLOYED"] = raw_features[
                "DAYS_EMPLOYED"
            ].replace(365243, np.nan)
        return engineer_features(raw_features)

    def prepare_input(
        self,
        applicant: dict[str, Any],
        model_version: str = "v1",
    ) -> pd.DataFrame:
        clean_data = {
            key: value
            for key, value in applicant.items()
            if key not in {"application_id", "name", "SK_ID_CURR", "TARGET"}
        }
        return self.prepare_frame(pd.DataFrame([clean_data]), model_version)

    def predict(
        self,
        applicant: dict[str, Any],
        model_version: str = "v1",
    ) -> tuple[float, Any]:
        bundle = self.get_model(model_version)
        df = self.prepare_input(applicant, model_version)
        X_processed = bundle.preprocessor.transform(df)
        probability = float(bundle.model.predict_proba(X_processed)[0][1])
        return probability, X_processed

    def explain(
        self,
        applicant: dict[str, Any],
        top_n: int = 5,
        model_version: str = "v1",
    ) -> list[dict[str, Any]]:
        bundle = self.get_model(model_version)
        df = self.prepare_input(applicant, model_version)
        X_processed = bundle.preprocessor.transform(df)
        shap_values = bundle.explainer.shap_values(X_processed)

        shap_array = np.asarray(shap_values)
        if shap_array.ndim != 2 or shap_array.shape != (
            1,
            len(bundle.feature_names),
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
                "feature": bundle.feature_names[index],
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