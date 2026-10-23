import pytest

from testing.analyze_thresholds import (
    calculate_threshold_metrics,
    select_candidate_threshold,
)


def test_calculate_threshold_metrics_counts_and_tradeoff() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.10, 0.40, 0.30, 0.80]

    high_cutoff, low_cutoff = calculate_threshold_metrics(
        labels,
        probabilities,
        [0.50, 0.20],
    )

    assert high_cutoff == {
        "threshold": 0.5,
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
        "true_positives": 1,
        "false_positives": 0,
        "true_negatives": 2,
        "false_negatives": 1,
        "predicted_positive_count": 1,
    }
    assert low_cutoff["recall"] == 1.0
    assert low_cutoff["true_positives"] == 2
    assert low_cutoff["false_positives"] == 1


def test_candidate_selection_uses_only_provided_diagnostic_metrics() -> None:
    candidate = select_candidate_threshold(
        [
            {"threshold": 0.05, "f1": 0.30},
            {"threshold": 0.10, "f1": 0.35},
            {"threshold": 0.20, "f1": 0.35},
        ]
    )

    assert candidate["threshold"] == 0.20


@pytest.mark.parametrize(
    ("labels", "probabilities", "thresholds"),
    [
        ([0, 1], [0.1], [0.5]),
        ([0, 2], [0.1, 0.9], [0.5]),
        ([0, 1], [0.1, 0.9], [1.1]),
    ],
)
def test_invalid_threshold_metric_inputs_raise(
    labels: list[int],
    probabilities: list[float],
    thresholds: list[float],
) -> None:
    with pytest.raises(ValueError):
        calculate_threshold_metrics(labels, probabilities, thresholds)
