"""Run secondary diagnostics against the existing saved baseline model.

This script reconstructs the notebook's stratified 80/20 split from
data/application_train.csv using test_size=0.20 and random_state=42. It uses
the saved model and preprocessor without fitting, retraining, or overwriting
either artifact. The original validation partition is split into two
reproducible, stratified diagnostic subsets. Both subsets remain secondary
diagnostics because the full original validation partition influenced
early-stopping selection in the training notebook.

Complete benchmark feature rows are used. TARGET is kept only as the label,
SK_ID_CURR is kept only to identify source records, and neither is passed to
the model. The script reproduces the notebook's ten engineered features,
applies the saved preprocessor, checks feature alignment, and reports ROC-AUC,
Average Precision, precision, recall, F1, and a confusion matrix at threshold
0.50. It writes JSON metrics, a readable Markdown report, and a CSV mapping
of each original validation row to its target label and diagnostic subset.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "application_train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgboost.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "baseline_preprocessor.joblib"
FEATURE_METADATA_PATH = PROJECT_ROOT / "models" / "baseline_feature_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "testing" / "reports"

ORIGINAL_VALIDATION_FRACTION = 0.20
ORIGINAL_SPLIT_SEED = 42
DIAGNOSTIC_HOLDOUT_FRACTION = 0.50
DIAGNOSTIC_SPLIT_SEED = 42
CLASSIFICATION_THRESHOLD = 0.50
EXPECTED_TRAINING_ROWS = 246_008
EXPECTED_VALIDATION_ROWS = 61_503
EXPECTED_RAW_FEATURES = 130
EXPECTED_ENGINEERED_FEATURES = [
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
]

EVALUATION_METHOD = """\
The script reads the existing Home Credit benchmark CSV, separates TARGET
from the feature matrix, excludes SK_ID_CURR from model inputs, and replaces
the DAYS_EMPLOYED sentinel 365243 with missing data as the training notebook
does. It recreates the notebook's stratified 80/20 split with random_state=42
and verifies the expected 246008 training and 61503 validation rows. The
saved preprocessor and saved XGBoost model are loaded as-is; no fitting,
retraining, or artifact rewriting occurs.

The original validation partition is split into a diagnostic-validation
subset and a secondary-holdout-diagnostic subset using a stratified 50/50
split with random_state=42. The same ten derived features from the notebook
are generated from complete benchmark rows. The saved preprocessor performs
the transformation and encoding. Both subsets are scored at a fixed
probability threshold of 0.50, and ROC-AUC, Average Precision, precision,
recall, F1, and a [TN, FP; FN, TP] confusion matrix are reported.

These are secondary diagnostics, not an independent final test estimate:
the original validation partition influenced early stopping and therefore
the selected model iteration. The emitted membership CSV identifies each
original validation row by CSV row offset and SK_ID_CURR, gives its TARGET
label, and records the diagnostic subset assignment.
"""


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the ten derived features defined in the training notebook."""
    data = frame.copy()
    data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25
    data["EMPLOYED_YEARS"] = -data["DAYS_EMPLOYED"] / 365.25

    def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator / denominator.replace(0, np.nan)

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
    data["INCOME_PER_CHILD"] = data["AMT_INCOME_TOTAL"] / (
        data["CNT_CHILDREN"] + 1
    )
    external_sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    data["EXT_SOURCE_MEAN"] = data[external_sources].mean(axis=1)
    data["EXT_SOURCE_MAX"] = data[external_sources].max(axis=1)
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_fingerprint(indices: pd.Index) -> str:
    ordered_indices = sorted(int(index) for index in indices)
    payload = ",".join(str(index) for index in ordered_indices).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def target_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().reindex([0, 1], fill_value=0)
    return {"0": int(counts.loc[0]), "1": int(counts.loc[1])}


def evaluate_partition(
    name: str,
    frame: pd.DataFrame,
    labels: pd.Series,
    preprocessor: Any,
    model: Any,
    expected_raw_columns: list[str],
    positive_class_index: int,
) -> dict[str, Any]:
    if not frame.index.equals(labels.index):
        raise AssertionError(f"{name}: feature rows and labels are misaligned.")
    if "TARGET" in frame.columns or "SK_ID_CURR" in frame.columns:
        raise AssertionError(f"{name}: target or source ID leaked into features.")

    engineered = engineer_features(frame)
    if engineered.columns.has_duplicates:
        raise AssertionError(f"{name}: duplicate columns after feature engineering.")
    if set(engineered.columns) != set(expected_raw_columns):
        missing = sorted(set(expected_raw_columns) - set(engineered.columns))
        unexpected = sorted(set(engineered.columns) - set(expected_raw_columns))
        raise AssertionError(
            f"{name}: raw feature schema mismatch; missing={missing}, "
            f"unexpected={unexpected}."
        )

    ordered = engineered.reindex(columns=expected_raw_columns)
    if ordered.columns.tolist() != expected_raw_columns:
        raise AssertionError(f"{name}: could not enforce preprocessor feature order.")

    transformed = preprocessor.transform(ordered)
    if transformed.shape[0] != len(labels):
        raise AssertionError(f"{name}: transformed row count does not match labels.")
    if transformed.shape[1] != model.n_features_in_:
        raise AssertionError(
            f"{name}: transformed width {transformed.shape[1]} does not match "
            f"model width {model.n_features_in_}."
        )

    probabilities = model.predict_proba(transformed)[:, positive_class_index]
    probability_series = pd.Series(probabilities, index=ordered.index)
    if not probability_series.index.equals(labels.index):
        raise AssertionError(f"{name}: predictions and labels are misaligned.")
    if len(probability_series) != len(labels) or not np.isfinite(probabilities).all():
        raise AssertionError(f"{name}: prediction count or values are invalid.")

    predicted_labels = (probability_series >= CLASSIFICATION_THRESHOLD).astype(int)
    matrix = confusion_matrix(labels, predicted_labels, labels=[0, 1])
    return {
        "record_count": int(len(labels)),
        "target_counts": target_counts(labels),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "roc_auc": float(roc_auc_score(labels, probability_series)),
        "average_precision": float(average_precision_score(labels, probability_series)),
        "precision": float(
            precision_score(labels, predicted_labels, zero_division=0)
        ),
        "recall": float(recall_score(labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(labels, predicted_labels, zero_division=0)),
        "confusion_matrix": {
            "labels": [0, 1],
            "layout": "[[TN, FP], [FN, TP]]",
            "values": matrix.astype(int).tolist(),
        },
        "source_row_index_sha256": row_fingerprint(labels.index),
    }


def build_membership(
    source: pd.DataFrame,
    validation_labels: pd.Series,
    diagnostic_index: pd.Index,
    holdout_index: pd.Index,
) -> pd.DataFrame:
    if not diagnostic_index.intersection(holdout_index).empty:
        raise AssertionError("Diagnostic subsets overlap.")
    if set(diagnostic_index) | set(holdout_index) != set(validation_labels.index):
        raise AssertionError("Diagnostic subsets do not cover original validation.")

    assignments = pd.Series(index=validation_labels.index, dtype="object")
    assignments.loc[diagnostic_index] = "diagnostic_validation"
    assignments.loc[holdout_index] = "secondary_holdout_diagnostic"
    membership = pd.DataFrame(
        {
            "source_row_index": validation_labels.index.astype(int),
            "SK_ID_CURR": source.loc[validation_labels.index, "SK_ID_CURR"].to_numpy(),
            "TARGET": validation_labels.to_numpy(dtype=int),
            "diagnostic_subset": assignments.loc[validation_labels.index].to_numpy(),
        }
    ).sort_values("source_row_index", ignore_index=True)
    if membership["SK_ID_CURR"].duplicated().any():
        raise AssertionError("Source IDs are not unique in validation membership.")
    return membership


def write_report(metrics: dict[str, Any], report_path: Path) -> None:
    partitions = metrics["diagnostic_partitions"]
    lines = [
        "# Saved Baseline Model: Secondary Diagnostic Evaluation",
        "",
        "> **Interpretation warning:** These results are secondary diagnostics,",
        "> not an independent final test estimate. The original validation",
        "> partition influenced early stopping and model-iteration selection.",
        "",
        "## Evaluation setup",
        "",
        f"- Source dataset: `{metrics['source_dataset']['path']}`",
        f"- Source SHA-256: `{metrics['source_dataset']['sha256']}`",
        f"- Dataset rows: {metrics['source_dataset']['row_count']:,}",
        f"- Original notebook split: {metrics['original_split']['training_rows']:,} "
        f"training / {metrics['original_split']['validation_rows']:,} validation",
        f"- Original split settings: {metrics['original_split']['settings']}",
        f"- Diagnostic partition split: "
        f"{metrics['diagnostic_split']['settings']}",
        f"- Classification threshold: {CLASSIFICATION_THRESHOLD:.2f}",
        f"- Transformed feature count: "
        f"{metrics['feature_alignment']['transformed_feature_count']}",
        "- Source IDs and `TARGET` are excluded from model features.",
        "",
        "## Partition membership and labels",
        "",
        "| Partition | Rows | TARGET=0 | TARGET=1 | Source-row-index SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for name, result in partitions.items():
        counts = result["target_counts"]
        lines.append(
            f"| {name} | {result['record_count']:,} | {counts['0']:,} | "
            f"{counts['1']:,} | `{result['source_row_index_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "Every original validation row and its target label are listed in "
            "`validation_partition_membership.csv`.",
            "",
            "## Metrics",
            "",
            "| Partition | ROC-AUC | Average Precision | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in partitions.items():
        lines.append(
            f"| {name} | {result['roc_auc']:.6f} | "
            f"{result['average_precision']:.6f} | {result['precision']:.6f} | "
            f"{result['recall']:.6f} | {result['f1']:.6f} |"
        )
    lines.extend(["", "### Confusion matrices at threshold 0.50", ""])
    for name, result in partitions.items():
        lines.extend(
            [
                f"**{name}** — labels `[0, 1]`, rows are actual and columns "
                "are predicted:",
                "",
                "```text",
                json.dumps(result["confusion_matrix"]["values"]),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Checks",
            "",
            *[
                f"- {check}: **{'PASS' if passed else 'FAIL'}**"
                for check, passed in metrics["checks"].items()
            ],
            "",
            "## Method implemented by `evaluate_saved_baseline.py`",
            "",
            "```text",
            EVALUATION_METHOD.rstrip(),
            "```",
            "",
            "## Limitations",
            "",
            "- This does not retrain or tune the saved model.",
            "- The diagnostic subsets are drawn from the validation partition",
            "  already used for early stopping; splitting it afterward does not",
            "  make either subset independent of that selection.",
            "- Metrics use the fixed 0.50 probability cutoff for class labels;",
            "  ROC-AUC and Average Precision evaluate ranking across thresholds.",
            "- These benchmark results do not establish production readiness or",
            "  real-world lending suitability.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    required_paths = [DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH, FEATURE_METADATA_PATH]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Required evaluation inputs are missing: {missing_paths}")

    source = pd.read_csv(DATA_PATH)
    required_columns = {"TARGET", "SK_ID_CURR", "DAYS_EMPLOYED"}
    if not required_columns.issubset(source.columns):
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(required_columns - set(source.columns))}"
        )
    if source["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR is not unique; cannot identify source rows safely.")

    labels = source["TARGET"].astype(int).copy()
    features = source.drop(columns=["TARGET", "SK_ID_CURR"]).copy()
    features["DAYS_EMPLOYED"] = features["DAYS_EMPLOYED"].replace(365243, np.nan)
    if "TARGET" in features or "SK_ID_CURR" in features:
        raise AssertionError("Target leakage or source ID is present in model inputs.")

    X_train, X_original_validation, y_train, y_original_validation = train_test_split(
        features,
        labels,
        test_size=ORIGINAL_VALIDATION_FRACTION,
        random_state=ORIGINAL_SPLIT_SEED,
        stratify=labels,
    )
    if len(X_train) != EXPECTED_TRAINING_ROWS:
        raise AssertionError(
            f"Reconstructed training rows ({len(X_train)}) do not match notebook "
            f"output ({EXPECTED_TRAINING_ROWS}). Stop; split was not reproduced."
        )
    if len(X_original_validation) != EXPECTED_VALIDATION_ROWS:
        raise AssertionError(
            f"Reconstructed validation rows ({len(X_original_validation)}) do not "
            f"match notebook output ({EXPECTED_VALIDATION_ROWS}). Stop; split was "
            "not reproduced."
        )
    if not X_train.index.intersection(X_original_validation.index).empty:
        raise AssertionError("Original training and validation rows overlap.")
    if not X_original_validation.index.equals(y_original_validation.index):
        raise AssertionError("Original validation features and labels are misaligned.")
    if "TARGET" in X_original_validation.columns:
        raise AssertionError("TARGET leaked into the reconstructed validation features.")

    X_diagnostic, X_holdout, y_diagnostic, y_holdout = train_test_split(
        X_original_validation,
        y_original_validation,
        test_size=DIAGNOSTIC_HOLDOUT_FRACTION,
        random_state=DIAGNOSTIC_SPLIT_SEED,
        stratify=y_original_validation,
    )
    if not X_diagnostic.index.intersection(X_holdout.index).empty:
        raise AssertionError("Diagnostic validation and holdout rows overlap.")
    if set(X_diagnostic.index) | set(X_holdout.index) != set(
        X_original_validation.index
    ):
        raise AssertionError("Diagnostic subsets do not reconstruct original validation.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))

    if "TARGET" in feature_metadata.get("numeric_features", []) or "TARGET" in (
        feature_metadata.get("categorical_features", [])
    ):
        raise AssertionError("Feature metadata includes TARGET.")
    if "SK_ID_CURR" in feature_metadata.get("numeric_features", []) or (
        "SK_ID_CURR" in feature_metadata.get("categorical_features", [])
    ):
        raise AssertionError("Feature metadata includes SK_ID_CURR.")

    transformer_columns = {
        name: list(columns)
        for name, _, columns in preprocessor.transformers_
        if name != "remainder"
    }
    expected_raw_columns = [
        column
        for name, _, columns in preprocessor.transformers_
        if name != "remainder"
        for column in columns
    ]
    metadata_columns = (
        feature_metadata["numeric_features"] + feature_metadata["categorical_features"]
    )
    if feature_metadata.get("engineered_features") != EXPECTED_ENGINEERED_FEATURES:
        raise AssertionError(
            "Script feature engineering does not match feature metadata."
        )
    if len(expected_raw_columns) != len(set(expected_raw_columns)):
        raise AssertionError("Saved preprocessor has duplicate raw feature columns.")
    if set(expected_raw_columns) != set(metadata_columns):
        raise AssertionError("Saved preprocessor and feature metadata schemas differ.")
    if len(expected_raw_columns) != EXPECTED_RAW_FEATURES:
        raise AssertionError(
            f"Expected {EXPECTED_RAW_FEATURES} raw model features, got "
            f"{len(expected_raw_columns)}."
        )
    if transformer_columns.get("num") != feature_metadata["numeric_features"]:
        raise AssertionError("Numeric feature order differs from feature metadata.")
    if transformer_columns.get("cat") != feature_metadata["categorical_features"]:
        raise AssertionError("Categorical feature order differs from feature metadata.")

    transformed_feature_names = preprocessor.get_feature_names_out()
    if len(transformed_feature_names) != model.n_features_in_:
        raise AssertionError(
            "Saved preprocessor output width does not match saved model input width."
        )
    model_classes = np.asarray(model.classes_)
    positive_indices = np.flatnonzero(model_classes == 1)
    if len(positive_indices) != 1:
        raise AssertionError(f"Could not locate positive class 1 in {model_classes}.")

    partitions = {
        "diagnostic_validation": evaluate_partition(
            "diagnostic_validation",
            X_diagnostic,
            y_diagnostic,
            preprocessor,
            model,
            expected_raw_columns,
            int(positive_indices[0]),
        ),
        "secondary_holdout_diagnostic": evaluate_partition(
            "secondary_holdout_diagnostic",
            X_holdout,
            y_holdout,
            preprocessor,
            model,
            expected_raw_columns,
            int(positive_indices[0]),
        ),
    }
    membership = build_membership(
        source,
        y_original_validation,
        X_diagnostic.index,
        X_holdout.index,
    )

    checks = {
        "reconstructed_split_matches_notebook_row_counts": True,
        "training_and_validation_partitions_are_disjoint": True,
        "diagnostic_subsets_are_disjoint_and_cover_validation": True,
        "TARGET_excluded_from_model_features": True,
        "SK_ID_CURR_excluded_from_model_features": True,
        "saved_preprocessor_schema_matches_feature_metadata": True,
        "transformed_feature_order_and_width_match_saved_model": True,
        "prediction_and_label_rows_are_aligned": True,
        "complete_benchmark_features_used_without_all_missing_fallback": True,
    }
    metrics = {
        "evaluation_type": "secondary diagnostic only; not an independent final test",
        "positive_class": 1,
        "positive_class_meaning": "TARGET=1 (payment difficulty/default)",
        "source_dataset": {
            "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(DATA_PATH),
            "row_count": int(len(source)),
            "target_counts": target_counts(labels),
        },
        "original_split": {
            "settings": {
                "test_size": ORIGINAL_VALIDATION_FRACTION,
                "random_state": ORIGINAL_SPLIT_SEED,
                "stratify": "TARGET",
            },
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_original_validation)),
            "training_target_counts": target_counts(y_train),
            "validation_target_counts": target_counts(y_original_validation),
        },
        "diagnostic_split": {
            "settings": {
                "validation_fraction": 1 - DIAGNOSTIC_HOLDOUT_FRACTION,
                "secondary_holdout_fraction": DIAGNOSTIC_HOLDOUT_FRACTION,
                "random_state": DIAGNOSTIC_SPLIT_SEED,
                "stratify": "TARGET",
            },
            "membership_file": "validation_partition_membership.csv",
        },
        "feature_alignment": {
            "raw_features_before_encoding": len(expected_raw_columns),
            "transformed_feature_count": len(transformed_feature_names),
            "model_input_feature_count": int(model.n_features_in_),
            "saved_model_classes": model_classes.tolist(),
            "preprocessor_numeric_features": len(feature_metadata["numeric_features"]),
            "preprocessor_categorical_features": len(
                feature_metadata["categorical_features"]
            ),
        },
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "diagnostic_partitions": partitions,
        "checks": checks,
        "runtime_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    membership.to_csv(
        REPORTS_DIR / "validation_partition_membership.csv",
        index=False,
    )
    write_report(metrics, REPORTS_DIR / "evaluation_report.md")

    print(f"Original split reproduced: {len(X_train):,} train / "
          f"{len(X_original_validation):,} validation")
    print(f"Diagnostic subsets: {len(X_diagnostic):,} / {len(X_holdout):,}")
    print(f"Reports written to: {REPORTS_DIR}")
    for name, result in partitions.items():
        print(
            f"{name}: ROC-AUC={result['roc_auc']:.6f}, "
            f"Average Precision={result['average_precision']:.6f}, "
            f"precision={result['precision']:.6f}, "
            f"recall={result['recall']:.6f}, F1={result['f1']:.6f}"
        )


if __name__ == "__main__":
    main()
