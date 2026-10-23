"""Analyze classification thresholds for the saved Drifting Oracle baseline.

This analysis reuses the saved model and preprocessor and the exact diagnostic
subsets produced by evaluate_saved_baseline.py. It does not fit or update any
model artifact. Threshold candidates are compared on diagnostic_validation;
the candidate with the highest F1 is carried to secondary_holdout_diagnostic
as a provisional consistency check. This is a technical comparison rule, not
a business objective or approved lending policy.

Both subsets came from the original validation partition, which influenced
early stopping. Neither subset is an independent final test set.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from testing.evaluate_saved_baseline import (
    DATA_PATH,
    DIAGNOSTIC_HOLDOUT_FRACTION,
    DIAGNOSTIC_SPLIT_SEED,
    EXPECTED_TRAINING_ROWS,
    EXPECTED_VALIDATION_ROWS,
    FEATURE_METADATA_PATH,
    MODEL_PATH,
    ORIGINAL_SPLIT_SEED,
    ORIGINAL_VALIDATION_FRACTION,
    PREPROCESSOR_PATH,
    REPORTS_DIR,
    engineer_features,
    row_fingerprint,
    sha256_file,
    target_counts,
)


EVALUATION_METRICS_PATH = REPORTS_DIR / "metrics.json"
MEMBERSHIP_PATH = REPORTS_DIR / "validation_partition_membership.csv"
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
OUTPUT_METRICS_PATH = REPORTS_DIR / "threshold_analysis_metrics.json"
OUTPUT_REPORT_PATH = REPORTS_DIR / "threshold_analysis_report.md"
OUTPUT_TABLE_PATH = REPORTS_DIR / "threshold_comparison.csv"
OUTPUT_CURVE_DATA_PATH = REPORTS_DIR / "precision_recall_curve.csv"
OUTPUT_CURVE_SVG_PATH = REPORTS_DIR / "precision_recall_curve.svg"


def calculate_threshold_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    """Calculate binary metrics and confusion counts for each cutoff."""
    y_true = np.asarray(list(labels), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores):
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays.")
    if len(y_true) == 0 or not np.isfinite(scores).all():
        raise ValueError("Labels must be nonempty and probabilities must be finite.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Labels must contain only binary values 0 and 1.")

    results = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}.")
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()
        results.append(
            {
                "threshold": float(threshold),
                "precision": float(
                    precision_score(y_true, predictions, zero_division=0)
                ),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "predicted_positive_count": int(tp + fp),
            }
        )
    return results


def select_candidate_threshold(
    diagnostic_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pick the diagnostic-only maximum-F1 candidate; prefer higher cutoff on ties."""
    if not diagnostic_results:
        raise ValueError("At least one diagnostic threshold result is required.")
    return max(
        diagnostic_results,
        key=lambda result: (result["f1"], result["threshold"]),
    )


def build_precision_recall_svg(
    precisions: np.ndarray,
    recalls: np.ndarray,
    threshold_results: list[dict[str, Any]],
    width: int = 900,
    height: int = 600,
) -> str:
    """Render a dependency-free SVG precision-recall curve with candidate markers."""
    margin_left, margin_right = 80, 35
    margin_top, margin_bottom = 55, 75
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def x_pos(recall: float) -> float:
        return margin_left + recall * plot_width

    def y_pos(precision: float) -> float:
        return margin_top + (1 - precision) * plot_height

    points = " ".join(
        f"{x_pos(float(recall)):.2f},{y_pos(float(precision)):.2f}"
        for recall, precision in zip(recalls, precisions)
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'viewBox="0 0 900 600" role="img" '
        'aria-labelledby="title description">',
        "<title id=\"title\">Precision-recall curve</title>",
        "<desc id=\"description\">Diagnostic validation precision-recall curve "
        "with candidate threshold markers.</desc>",
        "<rect width=\"100%\" height=\"100%\" fill=\"#fffdf7\"/>",
        '<text x="80" y="30" font-family="Arial, sans-serif" font-size="20" '
        'font-weight="bold" fill="#243746">Precision–Recall Curve</text>',
    ]
    for tick in np.linspace(0, 1, 6):
        x = x_pos(float(tick))
        y = y_pos(float(tick))
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
                f'y2="{margin_top + plot_height}" stroke="#e5e2d8"/>',
                f'<line x1="{margin_left}" y1="{y:.2f}" '
                f'x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e2d8"/>',
                f'<text x="{x:.2f}" y="{margin_top + plot_height + 22}" '
                'text-anchor="middle" font-family="Arial, sans-serif" '
                f'font-size="12" fill="#4c5960">{tick:.1f}</text>',
                f'<text x="{margin_left - 12}" y="{y + 4:.2f}" '
                'text-anchor="end" font-family="Arial, sans-serif" '
                f'font-size="12" fill="#4c5960">{tick:.1f}</text>',
            ]
        )
    elements.extend(
        [
            f'<polyline points="{points}" fill="none" stroke="#247f78" '
            'stroke-width="2.5" stroke-linejoin="round"/>',
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
            f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
            'stroke="#243746" stroke-width="1.5"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" '
            f'x2="{margin_left}" y2="{margin_top + plot_height}" '
            'stroke="#243746" stroke-width="1.5"/>',
            f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 20}" '
            'text-anchor="middle" font-family="Arial, sans-serif" '
            'font-size="14" fill="#243746">Recall</text>',
            f'<text transform="translate(22 {margin_top + plot_height / 2:.2f}) '
            'rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" '
            'font-size="14" fill="#243746">Precision</text>',
        ]
    )
    palette = ["#bf604c", "#b18732", "#6f65a8", "#27845f", "#496fa8", "#333333"]
    for index, result in enumerate(threshold_results):
        x = x_pos(result["recall"])
        y = y_pos(result["precision"])
        color = palette[index % len(palette)]
        label = html.escape(f"t={result['threshold']:.2f}")
        elements.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" '
                'stroke="#fffdf7" stroke-width="1.5"/>',
                f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" '
                'font-family="Arial, sans-serif" font-size="11" '
                f'fill="{color}">{label}</text>',
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements)


def _write_outputs(
    metrics: dict[str, Any],
    diagnostic_results: list[dict[str, Any]],
    curve_precisions: np.ndarray,
    curve_recalls: np.ndarray,
    curve_thresholds: np.ndarray,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostic_results).to_csv(OUTPUT_TABLE_PATH, index=False)
    curve_frame = pd.DataFrame(
        {
            "threshold": np.append(curve_thresholds, np.nan),
            "recall": curve_recalls,
            "precision": curve_precisions,
        }
    )
    curve_frame.to_csv(OUTPUT_CURVE_DATA_PATH, index=False)
    OUTPUT_CURVE_SVG_PATH.write_text(
        build_precision_recall_svg(
            curve_precisions, curve_recalls, diagnostic_results
        ),
        encoding="utf-8",
    )
    OUTPUT_METRICS_PATH.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    candidate = metrics["candidate_selection"]
    holdout = metrics["secondary_holdout_candidate_metrics"]
    lines = [
        "# Threshold Analysis — Existing Saved Baseline",
        "",
        "> **Diagnostic caveat:** Neither subset is an independent final test.",
        "> The original validation partition influenced early stopping and",
        "> model iteration selection.",
        "",
        "## Method",
        "",
        "- Reused the saved XGBoost model and preprocessor; neither was fitted,",
        "  tuned, overwritten, or regenerated.",
        "- Reused complete benchmark feature rows and existing diagnostic",
        "  partition membership. `TARGET` was kept outside model inputs and used",
        "  only for metrics.",
        "- Compared only the requested thresholds on `diagnostic_validation`.",
        "- Selected a candidate by the highest F1 among the requested values;",
        "  ties favor the higher threshold. This is a transparent technical",
        "  heuristic, **not** a business objective or an optimal policy.",
        "- Evaluated that candidate once on `secondary_holdout_diagnostic` as a",
        "  provisional consistency check, not to select or revise the candidate.",
        "",
        f"- Diagnostic validation records: "
        f"{metrics['partition_sizes']['diagnostic_validation']:,}",
        f"- Secondary holdout diagnostic records: "
        f"{metrics['partition_sizes']['secondary_holdout_diagnostic']:,}",
        f"- Diagnostic target counts: "
        f"{metrics['target_counts']['diagnostic_validation']}",
        f"- Positive class: TARGET=1 (payment difficulty/default)",
        f"- Selected candidate threshold: **{candidate['threshold']:.2f}**",
        f"- Candidate selection F1 (diagnostic validation): "
        f"{candidate['f1']:.6f}",
        "",
        "## Threshold comparison — diagnostic validation",
        "",
        "| Threshold | Precision | Recall | F1 | TP | FP | TN | FN | Predicted positive |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in diagnostic_results:
        marker = " **candidate**" if result["threshold"] == candidate["threshold"] else ""
        lines.append(
            f"| {result['threshold']:.2f}{marker} | {result['precision']:.6f} | "
            f"{result['recall']:.6f} | {result['f1']:.6f} | "
            f"{result['true_positives']:,} | {result['false_positives']:,} | "
            f"{result['true_negatives']:,} | {result['false_negatives']:,} | "
            f"{result['predicted_positive_count']:,} |"
        )
    lines.extend(
        [
            "",
            "The curve and marked candidate cutoffs are in "
            "`precision_recall_curve.svg`; underlying curve coordinates are in "
            "`precision_recall_curve.csv`.",
            "",
            "## Provisional consistency check on secondary holdout diagnostic",
            "",
            f"At the diagnostic-selected threshold **{holdout['threshold']:.2f}**: "
            f"precision **{holdout['precision']:.6f}**, "
            f"recall **{holdout['recall']:.6f}**, F1 **{holdout['f1']:.6f}**; "
            f"TP={holdout['true_positives']:,}, "
            f"FP={holdout['false_positives']:,}, "
            f"TN={holdout['true_negatives']:,}, "
            f"FN={holdout['false_negatives']:,}.",
            "",
            "This holdout result did not influence candidate selection. It is",
            "still not independent: it is part of the original validation",
            "partition that influenced early stopping.",
            "",
            "## Interpreting the trade-off",
            "",
            "Lowering the threshold labels more applicants as positive. This",
            "typically raises recall by catching more actual TARGET=1 cases,",
            "while also raising false positives and potentially lowering",
            "precision. Higher thresholds usually make fewer positive calls;",
            "precision may increase while recall falls. Select a deployment",
            "threshold only after defining the relative costs of missed defaults",
            "and false alarms and validating that objective on appropriate",
            "independent data. No threshold here is declared optimal.",
            "",
            "## Checks",
            "",
            *[
                f"- {check}: **{'PASS' if passed else 'FAIL'}**"
                for check, passed in metrics["checks"].items()
            ],
            "",
            "## Limitations",
            "",
            "- Diagnostic results are not independent final-test estimates.",
            "- The threshold-selection subset itself was part of the data used",
            "  for early-stopping selection, so its metrics may be optimistic.",
            "- The second subset is only a provisional consistency check.",
            "- The fixed 0.50 cutoff is retained as one comparator, not a",
            "  lending-policy recommendation.",
            "",
        ]
    )
    OUTPUT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    required_paths = [
        DATA_PATH,
        MODEL_PATH,
        PREPROCESSOR_PATH,
        FEATURE_METADATA_PATH,
        EVALUATION_METRICS_PATH,
        MEMBERSHIP_PATH,
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Threshold analysis requires existing evaluation inputs: {missing}"
        )

    saved_model_hash = sha256_file(MODEL_PATH)
    saved_preprocessor_hash = sha256_file(PREPROCESSOR_PATH)
    evaluation = json.loads(EVALUATION_METRICS_PATH.read_text(encoding="utf-8"))
    membership = pd.read_csv(MEMBERSHIP_PATH)
    source = pd.read_csv(DATA_PATH)
    source_hash = sha256_file(DATA_PATH)
    if source_hash != evaluation["source_dataset"]["sha256"]:
        raise ValueError("Dataset hash differs from the saved evaluation report.")
    if len(source) != evaluation["source_dataset"]["row_count"]:
        raise ValueError("Dataset row count differs from the saved evaluation report.")

    required_membership_columns = {
        "source_row_index",
        "SK_ID_CURR",
        "TARGET",
        "diagnostic_subset",
    }
    if not required_membership_columns.issubset(membership.columns):
        raise ValueError("Existing validation membership file has an invalid schema.")
    if membership["source_row_index"].duplicated().any():
        raise ValueError("Existing validation membership has duplicate source rows.")
    if set(membership["diagnostic_subset"]) != {
        "diagnostic_validation",
        "secondary_holdout_diagnostic",
    }:
        raise ValueError("Membership file has unexpected diagnostic partition labels.")

    labels = source["TARGET"].astype(int)
    features = source.drop(columns=["TARGET", "SK_ID_CURR"]).copy()
    features["DAYS_EMPLOYED"] = features["DAYS_EMPLOYED"].replace(365243, np.nan)
    X_train, X_validation, y_train, y_validation = train_test_split(
        features,
        labels,
        test_size=ORIGINAL_VALIDATION_FRACTION,
        random_state=ORIGINAL_SPLIT_SEED,
        stratify=labels,
    )
    X_diagnostic, X_holdout, y_diagnostic, y_holdout = train_test_split(
        X_validation,
        y_validation,
        test_size=DIAGNOSTIC_HOLDOUT_FRACTION,
        random_state=DIAGNOSTIC_SPLIT_SEED,
        stratify=y_validation,
    )
    if len(X_train) != EXPECTED_TRAINING_ROWS or len(X_validation) != EXPECTED_VALIDATION_ROWS:
        raise AssertionError("Original notebook split row counts did not reproduce.")
    if not X_train.index.intersection(X_validation.index).empty:
        raise AssertionError("Reconstructed training and validation rows overlap.")
    if not X_diagnostic.index.intersection(X_holdout.index).empty:
        raise AssertionError("Diagnostic validation and holdout rows overlap.")
    if set(X_diagnostic.index) | set(X_holdout.index) != set(X_validation.index):
        raise AssertionError("Diagnostic subsets do not cover original validation.")

    saved_membership = membership.set_index("source_row_index")
    for name, expected_index, expected_labels in (
        ("diagnostic_validation", X_diagnostic.index, y_diagnostic),
        ("secondary_holdout_diagnostic", X_holdout.index, y_holdout),
    ):
        actual_rows = saved_membership.loc[
            saved_membership["diagnostic_subset"] == name
        ]
        if set(actual_rows.index.astype(int)) != set(expected_index.astype(int)):
            raise AssertionError(f"{name} does not match saved partition membership.")
        expected_fingerprint = evaluation["diagnostic_partitions"][name][
            "source_row_index_sha256"
        ]
        if row_fingerprint(expected_index) != expected_fingerprint:
            raise AssertionError(f"{name} row fingerprint differs from saved report.")
        aligned = actual_rows.loc[expected_index]
        if not np.array_equal(
            aligned["TARGET"].to_numpy(dtype=int),
            expected_labels.to_numpy(dtype=int),
        ):
            raise AssertionError(f"{name} saved labels do not match source labels.")
        expected_ids = source.loc[expected_index, "SK_ID_CURR"].to_numpy()
        if not np.array_equal(aligned["SK_ID_CURR"].to_numpy(), expected_ids):
            raise AssertionError(f"{name} source applicant IDs do not match rows.")

    metadata = json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    if "TARGET" in features.columns or "SK_ID_CURR" in features.columns:
        raise AssertionError("TARGET or SK_ID_CURR leaked into model features.")
    if "TARGET" in metadata["numeric_features"] + metadata["categorical_features"]:
        raise AssertionError("Feature metadata contains TARGET.")
    expected_columns = [
        column
        for name, _, columns in preprocessor.transformers_
        if name != "remainder"
        for column in columns
    ]
    metadata_columns = metadata["numeric_features"] + metadata["categorical_features"]
    if (
        expected_columns != metadata_columns
        or len(expected_columns) != 130
    ):
        raise AssertionError("Saved preprocessor schema differs from expected metadata.")
    if len(preprocessor.get_feature_names_out()) != model.n_features_in_:
        raise AssertionError("Saved preprocessor output width differs from model input.")
    if metadata.get("engineered_features") != [
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
    ]:
        raise AssertionError("Feature engineering does not match saved metadata.")

    positive_indices = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive_indices) != 1:
        raise AssertionError("Saved model does not have exactly one positive class 1.")

    probabilities_by_partition = {}
    labels_by_partition = {}
    for name, frame, partition_labels in (
        ("diagnostic_validation", X_diagnostic, y_diagnostic),
        ("secondary_holdout_diagnostic", X_holdout, y_holdout),
    ):
        if "TARGET" in frame.columns or "SK_ID_CURR" in frame.columns:
            raise AssertionError(f"{name}: target or ID leaked into features.")
        engineered = engineer_features(frame)
        if set(engineered.columns) != set(expected_columns):
            raise AssertionError(f"{name}: input schema does not match preprocessor.")
        ordered = engineered.reindex(columns=expected_columns)
        transformed = preprocessor.transform(ordered)
        if transformed.shape != (len(partition_labels), model.n_features_in_):
            raise AssertionError(f"{name}: transformed feature matrix shape mismatch.")
        probabilities = model.predict_proba(transformed)[:, int(positive_indices[0])]
        if len(probabilities) != len(partition_labels) or not np.isfinite(probabilities).all():
            raise AssertionError(f"{name}: invalid probabilities or label alignment.")
        probabilities_by_partition[name] = probabilities
        labels_by_partition[name] = partition_labels.to_numpy(dtype=int)

    diagnostic_results = calculate_threshold_metrics(
        labels_by_partition["diagnostic_validation"],
        probabilities_by_partition["diagnostic_validation"],
        THRESHOLDS,
    )
    candidate = select_candidate_threshold(diagnostic_results)
    holdout_candidate_result = calculate_threshold_metrics(
        labels_by_partition["secondary_holdout_diagnostic"],
        probabilities_by_partition["secondary_holdout_diagnostic"],
        [candidate["threshold"]],
    )[0]

    curve_precision, curve_recall, curve_thresholds = precision_recall_curve(
        labels_by_partition["diagnostic_validation"],
        probabilities_by_partition["diagnostic_validation"],
    )
    average_precision = float(
        average_precision_score(
            labels_by_partition["diagnostic_validation"],
            probabilities_by_partition["diagnostic_validation"],
        )
    )
    holdout_candidate_result["partition"] = "secondary_holdout_diagnostic"
    checks = {
        "source_dataset_matches_existing_evaluation_hash": True,
        "partition_membership_matches_existing_evaluation": True,
        "partition_labels_and_source_ids_match_dataset": True,
        "TARGET_excluded_from_model_inputs": True,
        "saved_preprocessor_schema_matches_metadata": True,
        "transformed_feature_width_matches_model": True,
        "complete_benchmark_rows_used": True,
        "probability_and_label_alignment_verified": True,
        "diagnostic_only_threshold_candidate_selection": True,
        "model_artifact_unchanged": sha256_file(MODEL_PATH) == saved_model_hash,
        "preprocessor_artifact_unchanged": (
            sha256_file(PREPROCESSOR_PATH) == saved_preprocessor_hash
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"One or more threshold analysis checks failed: {checks}")

    metrics = {
        "evaluation_type": (
            "threshold analysis on secondary diagnostics; not an independent final test"
        ),
        "target": "TARGET=1 (payment difficulty/default)",
        "thresholds_compared": list(THRESHOLDS),
        "candidate_selection": {
            "method": "maximum F1 on diagnostic_validation; ties choose higher threshold",
            "business_objective_defined": False,
            "not_an_approved_decision_policy": True,
            **candidate,
        },
        "partition_sizes": {
            "diagnostic_validation": int(len(y_diagnostic)),
            "secondary_holdout_diagnostic": int(len(y_holdout)),
        },
        "target_counts": {
            "diagnostic_validation": target_counts(y_diagnostic),
            "secondary_holdout_diagnostic": target_counts(y_holdout),
        },
        "diagnostic_validation_average_precision": average_precision,
        "diagnostic_validation_threshold_metrics": diagnostic_results,
        "secondary_holdout_candidate_metrics": holdout_candidate_result,
        "files": {
            "report": OUTPUT_REPORT_PATH.name,
            "threshold_table": OUTPUT_TABLE_PATH.name,
            "precision_recall_curve_data": OUTPUT_CURVE_DATA_PATH.name,
            "precision_recall_curve_svg": OUTPUT_CURVE_SVG_PATH.name,
        },
        "checks": checks,
        "limitations": [
            "The original validation partition influenced early stopping.",
            "Neither diagnostic subset is an independent final test.",
            "The secondary subset is not used to select the candidate threshold.",
            "No business objective was supplied; no threshold is declared optimal.",
        ],
    }
    _write_outputs(
        metrics,
        diagnostic_results,
        curve_precision,
        curve_recall,
        curve_thresholds,
    )
    print(f"Candidate threshold (diagnostic-validation max F1): {candidate['threshold']:.2f}")
    print(
        f"Diagnostic validation: precision={candidate['precision']:.6f}, "
        f"recall={candidate['recall']:.6f}, F1={candidate['f1']:.6f}"
    )
    print(
        f"Secondary holdout consistency check: precision="
        f"{holdout_candidate_result['precision']:.6f}, "
        f"recall={holdout_candidate_result['recall']:.6f}, "
        f"F1={holdout_candidate_result['f1']:.6f}"
    )
    print(f"Threshold analysis reports written under {REPORTS_DIR}")


if __name__ == "__main__":
    main()
