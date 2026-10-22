import json

import pandas as pd

from testing.evaluate_saved_baseline import REPORTS_DIR, main


def test_saved_baseline_diagnostic_generates_aligned_reports() -> None:
    main()

    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    membership = pd.read_csv(
        REPORTS_DIR / "validation_partition_membership.csv"
    )

    assert metrics["original_split"]["training_rows"] == 246_008
    assert metrics["original_split"]["validation_rows"] == 61_503
    assert sum(
        partition["record_count"]
        for partition in metrics["diagnostic_partitions"].values()
    ) == 61_503
    assert len(membership) == 61_503
    assert membership["source_row_index"].is_unique
    assert membership["SK_ID_CURR"].is_unique
    assert set(membership["diagnostic_subset"]) == {
        "diagnostic_validation",
        "secondary_holdout_diagnostic",
    }
    assert all(metrics["checks"].values())
    assert (REPORTS_DIR / "evaluation_report.md").is_file()
