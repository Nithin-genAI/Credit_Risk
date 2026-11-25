import json
import sqlite3
from contextlib import closing

from backend.services.drift_monitor import (
    DriftMonitor,
    _categorical_metric,
    _feature_status,
    _load_reference,
    _numeric_metric,
)
from backend.services.prediction_service import prediction_service


def test_v1_reference_profiles_match_preprocessor_schema_and_types():
    reference = _load_reference("v1")
    expected_features = set(prediction_service.get_model("v1").input_columns)

    assert set(reference) == expected_features
    assert reference["EXT_SOURCE_MEAN"]["type"] == "numeric"
    assert reference["NAME_CONTRACT_TYPE"]["type"] == "categorical"
    assert len(reference["NAME_CONTRACT_TYPE"]["frequencies"]) < 100


def test_numeric_psi_and_missing_rate_detect_material_shift():
    metric = _numeric_metric(
        "AMT_CREDIT",
        {
            "type": "numeric",
            "missing_rate": 0.0,
            "quantiles": {
                "0": 0,
                "0.25": 10,
                "0.5": 20,
                "0.75": 30,
                "1": 40,
            },
        },
        [100.0] * 50 + [None] * 50,
        0.0,
    )

    assert metric["psi"] >= 0.2
    assert metric["missing_rate_change"] == 0.5
    assert metric["status"] == "Oracle Drifted"
    assert metric["ks_statistic"] is not None
    assert metric["current_distribution"]["above 40"] == 0.5


def test_categorical_metrics_include_psi_and_chi_square():
    definition = {
        "type": "categorical",
        "missing_rate": 0.0,
        "frequencies": {"Cash loans": 0.5, "Revolving loans": 0.5},
    }
    stable = _categorical_metric(
        "NAME_CONTRACT_TYPE",
        definition,
        ["Cash loans"] * 50 + ["Revolving loans"] * 50,
    )
    metric = _categorical_metric(
        "NAME_CONTRACT_TYPE",
        definition,
        ["Cash loans"] * 100,
    )

    assert stable["psi"] == 0
    assert stable["status"] == "No Significant Drift"
    assert metric["psi"] >= 0.2
    assert metric["chi_square_statistic"] > 0
    assert metric["chi_square_p_value"] < 0.01
    assert metric["status"] == "Oracle Drifted"
    assert _feature_status(0.15, 0.0) == "Moderate Drift"
    assert _feature_status(0.0, -0.1) == "Oracle Drifted"


def test_prediction_log_stores_only_model_features_and_outcomes(tmp_path):
    database_path = tmp_path / "monitor.sqlite3"
    monitor = DriftMonitor(database_path)
    entries = [
        {
            "prediction_id": "00000000-0000-0000-0000-000000000001",
            "model_version": "v1",
            "probability": 0.8,
            "feature_values": {
                "AMT_CREDIT": 500000,
                "application_id": "private-application",
                "name": "Private Name",
                "SK_ID_CURR": 123,
                "TARGET": 1,
            },
        }
    ]

    assert monitor.record_predictions(entries)
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT model_version, probability, feature_values "
            "FROM predictions"
        ).fetchone()
    assert row[:2] == ("v1", 0.8)
    features = json.loads(row[2])
    assert features == {"AMT_CREDIT": 500000}

    outcome_result = monitor.record_outcomes(
        [
            ("00000000-0000-0000-0000-000000000001", 1),
            ("00000000-0000-0000-0000-000000000002", 0),
        ]
    )
    assert outcome_result == {
        "updated_count": 1,
        "unknown_prediction_ids": ["00000000-0000-0000-0000-000000000002"],
    }


def test_logging_storage_failure_is_reported_without_raising(tmp_path):
    unusable_path = tmp_path / "not_a_database"
    unusable_path.mkdir()
    monitor = DriftMonitor(unusable_path)

    assert not monitor.record_predictions(
        [
            {
                "model_version": "v1",
                "probability": 0.5,
                "feature_values": {"AMT_CREDIT": 100000},
            }
        ]
    )


def test_performance_metrics_use_labeled_predictions_and_versions_are_isolated(
    tmp_path,
):
    monitor = DriftMonitor(tmp_path / "monitor.sqlite3")
    v1_ids = [
        f"00000000-0000-0000-0000-{index:012d}"
        for index in range(1, 5)
    ]
    entries = []
    for prediction_id, probability in zip(
        v1_ids,
        (0.1, 0.8, 0.2, 0.9),
    ):
        entries.append(
            {
                "prediction_id": prediction_id,
                "model_version": "v1",
                "probability": probability,
                "feature_values": {"AMT_CREDIT": 250000},
            }
        )
    entries.append(
        {
            "prediction_id": "00000000-0000-0000-0000-000000000005",
            "model_version": "v2",
            "probability": 0.4,
            "feature_values": {"AMT_CREDIT": 250000},
        }
    )
    assert monitor.record_predictions(entries)
    assert monitor.record_outcomes(
        [
            (v1_ids[0], 0),
            (v1_ids[1], 1),
            (v1_ids[2], 0),
            (v1_ids[3], 1),
        ]
    )["updated_count"] == 4

    v1_report = monitor.report("v1")
    v2_report = monitor.report("v2")

    assert v1_report["sample_count"] == 4
    assert v1_report["status"] == "Insufficient Data"
    assert v1_report["performance"]["sample_count"] == 4
    assert v1_report["performance"]["roc_auc"] == 1.0
    assert v2_report["sample_count"] == 1
    assert v2_report["performance"]["sample_count"] == 0
    assert v2_report["model_version"] == "v2"
