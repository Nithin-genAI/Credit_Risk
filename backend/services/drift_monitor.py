"""Local prediction logging and reference-based model drift reports."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import uuid
from contextlib import contextmanager
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare, kstest
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score

from backend.services.prediction_service import prediction_service
from backend.settings import (
    CLASSIFICATION_THRESHOLD,
    DRIFT_MAX_MONITORING_ROWS,
    DRIFT_MIN_SAMPLE_SIZE,
    DRIFT_MISSING_CRITICAL_DELTA,
    DRIFT_MISSING_MODERATE_DELTA,
    DRIFT_PSI_CRITICAL_THRESHOLD,
    DRIFT_PSI_MODERATE_THRESHOLD,
    DRIFT_RETENTION_DAYS,
    DRIFT_WINDOW_DAYS,
    MONITORING_DB_PATH,
    PROJECT_PATH,
    V2_CLASSIFICATION_THRESHOLD,
)


logger = logging.getLogger(__name__)

REFERENCE_PATHS = {
    "v1": PROJECT_PATH / "models" / "reference_distribution.json",
    "v2": PROJECT_PATH / "fine_tuning" / "artifacts" / "v2_reference_state.json",
}
PROBABILITY_REFERENCE_PATHS = {
    "v1": PROJECT_PATH / "models" / "baseline_reference_metrics.json",
    "v2": PROJECT_PATH / "fine_tuning" / "artifacts" / "v2_oof_predictions.csv",
}
MODEL_THRESHOLDS = {
    "v1": CLASSIFICATION_THRESHOLD,
    "v2": V2_CLASSIFICATION_THRESHOLD,
}
EPSILON = 1e-6


class DriftStorageError(RuntimeError):
    """Monitoring storage could not be read or written."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _quantile_points(definition: dict[str, Any]) -> list[tuple[float, float]]:
    raw_quantiles = definition.get("quantiles")
    points: list[tuple[float, float]] = []
    if isinstance(raw_quantiles, dict):
        for key, value in raw_quantiles.items():
            try:
                quantile = float(key)
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if 0 <= quantile <= 1 and math.isfinite(numeric_value):
                points.append((quantile, numeric_value))
    else:
        for key, quantile in (
            ("min", 0.0),
            ("q01", 0.01),
            ("q05", 0.05),
            ("q25", 0.25),
            ("q50", 0.5),
            ("q75", 0.75),
            ("q95", 0.95),
            ("q99", 0.99),
            ("max", 1.0),
        ):
            if key in definition:
                try:
                    value = float(definition[key])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    points.append((quantile, value))
    return sorted(points)


@lru_cache(maxsize=2)
def _load_reference(model_version: str) -> dict[str, dict[str, Any]]:
    path = REFERENCE_PATHS[model_version]
    reference = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict):
        raise ValueError(f"Reference state for {model_version} must be an object.")
    return reference


@lru_cache(maxsize=2)
def _load_probability_reference(
    model_version: str,
) -> list[tuple[float, float]]:
    path = PROBABILITY_REFERENCE_PATHS[model_version]
    if model_version == "v1":
        metrics = json.loads(path.read_text(encoding="utf-8"))
        points = [
            (0.0, float(metrics["min"])),
            (0.25, float(metrics["p25"])),
            (0.5, float(metrics["p50"])),
            (0.75, float(metrics["p75"])),
            (1.0, float(metrics["max"])),
        ]
    else:
        probabilities = pd.read_csv(path, usecols=["probability"])["probability"]
        points = [
            (quantile, float(probabilities.quantile(quantile)))
            for quantile in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
        ]
    if len(points) < 2:
        raise ValueError(f"Probability reference for {model_version} is incomplete.")
    return points


def _interpolated_cdf(points: list[tuple[float, float]], values: Any) -> np.ndarray:
    by_value: dict[float, float] = {}
    for quantile, value in points:
        by_value[value] = max(quantile, by_value.get(value, 0.0))
    ordered = sorted(by_value.items())
    x_values = np.asarray([point[0] for point in ordered], dtype=float)
    y_values = np.asarray([point[1] for point in ordered], dtype=float)
    return np.interp(
        np.asarray(values, dtype=float),
        x_values,
        y_values,
        left=0.0,
        right=1.0,
    )


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    reference = np.clip(reference.astype(float), EPSILON, None)
    current = np.clip(current.astype(float), EPSILON, None)
    reference /= reference.sum()
    current /= current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def _feature_status(psi_value: float, missing_delta: float | None) -> str:
    abs_missing_delta = abs(missing_delta) if missing_delta is not None else 0.0
    if (
        psi_value >= DRIFT_PSI_CRITICAL_THRESHOLD
        or abs_missing_delta >= DRIFT_MISSING_CRITICAL_DELTA
    ):
        return "Oracle Drifted"
    if (
        psi_value >= DRIFT_PSI_MODERATE_THRESHOLD
        or abs_missing_delta >= DRIFT_MISSING_MODERATE_DELTA
    ):
        return "Moderate Drift"
    return "No Significant Drift"


def _numeric_metric(
    name: str,
    definition: dict[str, Any],
    current_values: list[Any],
    reference_missing_rate: float,
) -> dict[str, Any]:
    points = _quantile_points(definition)
    if len(points) < 2:
        return {
            "feature": name,
            "feature_type": "numeric",
            "status": "Reference Unavailable",
            "psi": None,
            "ks_statistic": None,
            "ks_p_value": None,
            "reference_distribution": {},
            "current_distribution": {},
            "reference_missing_rate": reference_missing_rate,
            "current_missing_rate": None,
            "missing_rate_change": None,
        }
    if not current_values:
        return {
            "feature": name,
            "feature_type": "numeric",
            "status": "Insufficient Data",
            "psi": None,
            "ks_statistic": None,
            "ks_p_value": None,
            "reference_distribution": {},
            "current_distribution": {},
            "reference_missing_rate": reference_missing_rate,
            "current_missing_rate": None,
            "missing_rate_change": None,
        }

    numeric = pd.to_numeric(pd.Series(current_values), errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    finite_mask = np.isfinite(numeric)
    clean = numeric[finite_mask]
    current_missing_rate = float(1.0 - finite_mask.mean()) if len(numeric) else 0.0

    quantile_edges: dict[float, float] = {}
    for quantile, value in points:
        if 0 <= quantile <= 1:
            quantile_edges[value] = max(quantile, quantile_edges.get(value, 0.0))
    edges = sorted(quantile_edges.items())
    edge_values = np.asarray([item[0] for item in edges], dtype=float)
    cumulative = np.asarray([item[1] for item in edges], dtype=float)
    ref_nonmissing = np.diff(np.concatenate(([0.0], cumulative, [1.0])))
    if len(clean):
        bin_indexes = np.searchsorted(edge_values, clean, side="right")
        observed_counts = np.bincount(bin_indexes, minlength=len(ref_nonmissing))
        current_nonmissing = observed_counts / len(numeric)
    else:
        observed_counts = np.zeros(len(ref_nonmissing), dtype=int)
        current_nonmissing = np.zeros(len(ref_nonmissing), dtype=float)

    reference_bins = ref_nonmissing * (1.0 - reference_missing_rate)
    current_bins = current_nonmissing
    reference_values = np.append(reference_bins, reference_missing_rate)
    current_values_with_missing = np.append(current_bins, current_missing_rate)
    labels = []
    for bin_index in range(len(ref_nonmissing)):
        if bin_index == 0:
            labels.append(f"below {edge_values[0]:g}")
        elif bin_index == len(edge_values):
            labels.append(f"above {edge_values[-1]:g}")
        else:
            labels.append(
                f"{edge_values[bin_index - 1]:g}-{edge_values[bin_index]:g}"
            )
    labels.append("missing")
    reference_distribution = {
        label: float(value)
        for label, value in zip(labels, reference_values)
    }
    current_distribution = {
        label: float(value)
        for label, value in zip(labels, current_values_with_missing)
    }
    psi_value = _psi(reference_values, current_values_with_missing)

    ks_statistic = None
    ks_p_value = None
    if len(clean):
        try:
            result = kstest(
                clean,
                lambda sample: _interpolated_cdf(points, sample),
                method="auto",
            )
            ks_statistic = float(result.statistic)
            ks_p_value = float(result.pvalue)
        except (ValueError, FloatingPointError):
            logger.warning("numeric_ks_failed feature=%s", name)

    missing_delta = current_missing_rate - reference_missing_rate
    return {
        "feature": name,
        "feature_type": "numeric",
        "status": _feature_status(psi_value, missing_delta),
        "psi": psi_value,
        "ks_statistic": ks_statistic,
        "ks_p_value": ks_p_value,
        "ks_method": "approximate_from_saved_reference_quantiles",
        "reference_distribution": reference_distribution,
        "current_distribution": current_distribution,
        "reference_missing_rate": reference_missing_rate,
        "current_missing_rate": current_missing_rate,
        "missing_rate_change": missing_delta,
    }


def _categorical_metric(
    name: str,
    definition: dict[str, Any],
    current_values: list[Any],
) -> dict[str, Any]:
    reference_frequencies = {
        str(key): float(value)
        for key, value in definition.get("frequencies", {}).items()
    }
    if not reference_frequencies:
        return {
            "feature": name,
            "feature_type": "categorical",
            "status": "Reference Unavailable",
            "psi": None,
            "chi_square_statistic": None,
            "chi_square_p_value": None,
            "reference_distribution": {},
            "current_distribution": {},
            "reference_missing_rate": definition.get("missing_rate"),
            "current_missing_rate": None,
            "missing_rate_change": None,
        }
    if not current_values:
        return {
            "feature": name,
            "feature_type": "categorical",
            "status": "Insufficient Data",
            "psi": None,
            "chi_square_statistic": None,
            "chi_square_p_value": None,
            "reference_distribution": reference_frequencies,
            "current_distribution": {},
            "reference_missing_rate": definition.get("missing_rate"),
            "current_missing_rate": None,
            "missing_rate_change": None,
        }

    current_categories = [
        "__MISSING__" if value is None or pd.isna(value) else str(value)
        for value in current_values
    ]
    categories = sorted(set(reference_frequencies) | set(current_categories))
    ref = np.asarray(
        [reference_frequencies.get(category, 0.0) for category in categories],
        dtype=float,
    )
    ref_missing_rate = float(
        definition.get("missing_rate", reference_frequencies.get("__MISSING__", 0.0))
    )
    if "__MISSING__" not in reference_frequencies and ref_missing_rate:
        ref[categories.index("__MISSING__")] = ref_missing_rate
    current_counts = pd.Series(current_categories).value_counts()
    observed = np.asarray(
        [int(current_counts.get(category, 0)) for category in categories],
        dtype=float,
    )
    sample_count = len(current_categories)
    current = observed / sample_count if sample_count else np.zeros(len(categories))
    psi_value = _psi(ref, current) if sample_count else 0.0
    current_missing_rate = (
        current_categories.count("__MISSING__") / sample_count if sample_count else 0.0
    )

    chi_square_statistic = None
    chi_square_p_value = None
    if sample_count and len(categories) > 1:
        expected_probabilities = np.clip(ref, EPSILON, None)
        expected_probabilities /= expected_probabilities.sum()
        try:
            result = chisquare(
                observed,
                f_exp=expected_probabilities * sample_count,
            )
            chi_square_statistic = float(result.statistic)
            chi_square_p_value = float(result.pvalue)
        except (ValueError, FloatingPointError):
            logger.warning("categorical_chi_square_failed feature=%s", name)

    missing_delta = current_missing_rate - ref_missing_rate
    return {
        "feature": name,
        "feature_type": "categorical",
        "status": _feature_status(psi_value, missing_delta),
        "psi": psi_value,
        "chi_square_statistic": chi_square_statistic,
        "chi_square_p_value": chi_square_p_value,
        "reference_distribution": {
            category: float(value)
            for category, value in zip(categories, ref)
        },
        "current_distribution": {
            category: float(value)
            for category, value in zip(categories, current)
        },
        "reference_missing_rate": ref_missing_rate,
        "current_missing_rate": current_missing_rate,
        "missing_rate_change": missing_delta,
    }


class DriftMonitor:
    def __init__(self, database_path: Path | str = MONITORING_DB_PATH):
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                predicted_at TEXT NOT NULL,
                model_version TEXT NOT NULL CHECK (model_version IN ('v1', 'v2')),
                probability REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
                feature_values TEXT NOT NULL,
                batch_id TEXT,
                actual_outcome INTEGER CHECK (
                    actual_outcome IS NULL OR actual_outcome IN (0, 1)
                )
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_model_time
                ON predictions(model_version, predicted_at);
            """
        )
        return connection

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record_predictions(self, entries: list[dict[str, Any]]) -> bool:
        if not entries:
            return True
        timestamp = _utc_now().isoformat(timespec="seconds")
        cutoff = (_utc_now() - timedelta(days=DRIFT_RETENTION_DAYS)).isoformat(
            timespec="seconds"
        )
        rows = []
        try:
            for entry in entries:
                version = prediction_service.normalize_version(entry["model_version"])
                probability = float(entry["probability"])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Prediction probabilities must be between 0 and 1.")
                allowed_features = set(
                    prediction_service.get_model(version).input_columns
                )
                features = {
                    str(key): _json_safe_value(value)
                    for key, value in entry["feature_values"].items()
                    if key in allowed_features
                }
                rows.append(
                    (
                        str(entry.get("prediction_id") or uuid.uuid4()),
                        timestamp,
                        version,
                        probability,
                        json.dumps(features, separators=(",", ":"), allow_nan=False),
                        entry.get("batch_id"),
                    )
                )
            with self._connection() as connection:
                connection.executemany(
                    """
                    INSERT INTO predictions (
                        prediction_id, predicted_at, model_version,
                        probability, feature_values, batch_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                connection.execute(
                    "DELETE FROM predictions WHERE predicted_at < ?",
                    (cutoff,),
                )
            return True
        except (
            sqlite3.Error,
            OSError,
            TypeError,
            ValueError,
            OverflowError,
            KeyError,
        ) as exc:
            logger.warning(
                "prediction_monitoring_log_failed error_type=%s",
                type(exc).__name__,
            )
            return False

    def record_outcomes(
        self,
        outcomes: list[tuple[str, int]],
    ) -> dict[str, Any]:
        try:
            with self._connection() as connection:
                identifiers = [prediction_id for prediction_id, _ in outcomes]
                placeholders = ",".join("?" for _ in identifiers)
                existing = {
                    row["prediction_id"]
                    for row in connection.execute(
                        f"SELECT prediction_id FROM predictions "
                        f"WHERE prediction_id IN ({placeholders})",
                        identifiers,
                    )
                }
                updates = [
                    (actual_outcome, prediction_id)
                    for prediction_id, actual_outcome in outcomes
                    if prediction_id in existing
                ]
                connection.executemany(
                    "UPDATE predictions SET actual_outcome = ? "
                    "WHERE prediction_id = ?",
                    updates,
                )
            return {
                "updated_count": len(updates),
                "unknown_prediction_ids": [
                    prediction_id
                    for prediction_id in identifiers
                    if prediction_id not in existing
                ],
            }
        except (sqlite3.Error, OSError) as exc:
            raise DriftStorageError("Could not save prediction outcomes.") from exc

    def _fetch_predictions(
        self,
        model_version: str,
        window_days: int,
    ) -> list[sqlite3.Row]:
        since = (_utc_now() - timedelta(days=window_days)).isoformat(
            timespec="seconds"
        )
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT prediction_id, predicted_at, model_version,
                           probability, feature_values, actual_outcome
                    FROM predictions
                    WHERE model_version = ? AND predicted_at >= ?
                    ORDER BY predicted_at DESC
                    LIMIT ?
                    """,
                    (model_version, since, DRIFT_MAX_MONITORING_ROWS),
                ).fetchall()
            return list(reversed(rows))
        except (sqlite3.Error, OSError) as exc:
            raise DriftStorageError("Could not read prediction monitoring data.") from exc

    @staticmethod
    def _performance_report(rows: list[sqlite3.Row]) -> dict[str, Any]:
        labeled = [row for row in rows if row["actual_outcome"] is not None]
        if not labeled:
            return {
                "sample_count": 0,
                "actual_default_rate": None,
                "roc_auc": None,
                "brier_score": None,
                "precision": None,
                "recall": None,
                "note": "No actual outcomes have been recorded for this window.",
            }
        actual = np.asarray([int(row["actual_outcome"]) for row in labeled])
        probabilities = np.asarray([float(row["probability"]) for row in labeled])
        predicted = probabilities >= np.asarray(
            [MODEL_THRESHOLDS[row["model_version"]] for row in labeled]
        )
        return {
            "sample_count": len(labeled),
            "actual_default_rate": float(actual.mean()),
            "roc_auc": (
                float(roc_auc_score(actual, probabilities))
                if len(np.unique(actual)) == 2
                else None
            ),
            "brier_score": float(brier_score_loss(actual, probabilities)),
            "precision": float(precision_score(actual, predicted, zero_division=0)),
            "recall": float(recall_score(actual, predicted, zero_division=0)),
            "note": None if len(np.unique(actual)) == 2 else (
                "ROC-AUC requires both outcome classes in the labeled window."
            ),
        }

    def analyze(
        self,
        model_version: str,
        window_days: int | None = None,
    ) -> dict[str, Any]:
        version = prediction_service.normalize_version(model_version)
        days = DRIFT_WINDOW_DAYS if window_days is None else int(window_days)
        if not 1 <= days <= 365:
            raise ValueError("window_days must be between 1 and 365.")

        rows = self._fetch_predictions(version, days)
        records = [
            {
                **json.loads(row["feature_values"]),
                "__probability__": float(row["probability"]),
            }
            for row in rows
        ]
        reference = _load_reference(version)
        sample_count = len(rows)
        feature_metrics: list[dict[str, Any]] = []
        matched_features = 0

        for feature, definition in reference.items():
            if not isinstance(definition, dict):
                continue
            current_values = [record.get(feature) for record in records]
            matched_features += 1
            feature_type = definition.get("type")
            if feature_type == "categorical" or "frequencies" in definition:
                metric = _categorical_metric(feature, definition, current_values)
            else:
                reference_rows = max(
                    (
                        float(item.get("count", 0))
                        for item in reference.values()
                        if isinstance(item, dict)
                    ),
                    default=0.0,
                )
                missing_rate = definition.get("missing_rate")
                if missing_rate is None:
                    missing_rate = (
                        1.0 - float(definition.get("count", 0)) / reference_rows
                        if reference_rows
                        else 0.0
                    )
                metric = _numeric_metric(
                    feature,
                    definition,
                    current_values,
                    min(max(float(missing_rate), 0.0), 1.0),
                )
            feature_metrics.append(metric)

        probability_points = _load_probability_reference(version)
        probability_metric = _numeric_metric(
            "prediction_probability",
            {"type": "numeric", "quantiles": {
                str(quantile): value for quantile, value in probability_points
            }},
            [record["__probability__"] for record in records],
            0.0,
        )
        probability_metric["feature_type"] = "prediction_probability"
        feature_metrics.append(probability_metric)
        feature_metrics.sort(
            key=lambda item: (
                item["psi"] is None,
                -(item["psi"] or 0.0),
            )
        )

        if sample_count < DRIFT_MIN_SAMPLE_SIZE:
            overall_status = "Insufficient Data"
            reasons = [
                f"At least {DRIFT_MIN_SAMPLE_SIZE} predictions are required "
                "before drift can be classified."
            ]
        else:
            severities = {metric["status"] for metric in feature_metrics}
            if "Oracle Drifted" in severities:
                overall_status = "Oracle Drifted"
                reasons = ["At least one monitored feature exceeds a critical threshold."]
            elif "Moderate Drift" in severities:
                overall_status = "Moderate Drift"
                reasons = ["At least one monitored feature exceeds a moderate threshold."]
            else:
                overall_status = "No Significant Drift"
                reasons = ["All comparable metrics are below configured thresholds."]

        times = [row["predicted_at"] for row in rows]
        return {
            "model_version": version,
            "status": overall_status,
            "sample_count": sample_count,
            "minimum_sample_size": DRIFT_MIN_SAMPLE_SIZE,
            "window_days": days,
            "window_start": times[0] if times else None,
            "window_end": times[-1] if times else None,
            "reference_feature_count": len(reference),
            "compared_feature_count": matched_features,
            "reference_coverage": (
                round(matched_features / len(reference), 4) if reference else 0.0
            ),
            "thresholds": {
                "psi_moderate": DRIFT_PSI_MODERATE_THRESHOLD,
                "psi_critical": DRIFT_PSI_CRITICAL_THRESHOLD,
                "missing_rate_moderate_delta": DRIFT_MISSING_MODERATE_DELTA,
                "missing_rate_critical_delta": DRIFT_MISSING_CRITICAL_DELTA,
            },
            "reasons": reasons,
            "feature_metrics": feature_metrics,
            "performance": self._performance_report(rows),
        }

    def status(self, model_version: str) -> dict[str, Any]:
        report = self.analyze(model_version)
        return {
            key: value
            for key, value in report.items()
            if key != "feature_metrics"
        }

    def report(self, model_version: str) -> dict[str, Any]:
        return self.analyze(model_version)


drift_monitor = DriftMonitor()
