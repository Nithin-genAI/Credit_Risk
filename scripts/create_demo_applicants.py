"""Create reproducible Home Credit benchmark records for UI demonstrations.

The sampled records are benchmark data, not verified real-world bank applications.
TARGET is intentionally never loaded or written to the assessment payload.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "application_train.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "demo_applicants.csv"
SAMPLE_SIZE = 30
RANDOM_SEED = 42

SOURCE_FIELDS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "NAME_CONTRACT_TYPE",
]
REQUIRED_FIELDS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Home Credit source file not found: {SOURCE_PATH}")

    source = pd.read_csv(SOURCE_PATH, usecols=SOURCE_FIELDS)
    eligible = source.dropna(subset=REQUIRED_FIELDS)
    if len(eligible) < SAMPLE_SIZE:
        raise ValueError(
            f"Need {SAMPLE_SIZE} eligible records but found {len(eligible)}."
        )

    sample = eligible.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(
        drop=True
    )
    sample.insert(
        0,
        "application_id",
        [f"DEMO-{index:04d}" for index in range(1, SAMPLE_SIZE + 1)],
    )
    sample.insert(
        1,
        "name",
        [f"Benchmark Applicant {index:02d}" for index in range(1, SAMPLE_SIZE + 1)],
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {OUTPUT_PATH} with {len(sample)} sampled rows.")
    print("TARGET and source applicant IDs are excluded.")


if __name__ == "__main__":
    main()
