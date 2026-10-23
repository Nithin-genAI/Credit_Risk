"""Create two consecutive 100-row batches from the saved application test CSV."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "application_test.csv"
FIRST_BATCH_PATH = PROJECT_ROOT / "data" / "application_test1.csv"
SECOND_BATCH_PATH = PROJECT_ROOT / "data" / "application_test2.csv"
BATCH_SIZE = 100


def main() -> None:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Application test file not found: {SOURCE_PATH}")

    rows = pd.read_csv(SOURCE_PATH, nrows=2 * BATCH_SIZE)
    if len(rows) != 2 * BATCH_SIZE:
        raise ValueError(
            f"Expected at least {2 * BATCH_SIZE} rows in {SOURCE_PATH}, "
            f"but found only {len(rows)}."
        )
    if "TARGET" in rows.columns:
        raise ValueError("TARGET must not be included in inference batch files.")

    FIRST_BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.iloc[:BATCH_SIZE].to_csv(FIRST_BATCH_PATH, index=False)
    rows.iloc[BATCH_SIZE : 2 * BATCH_SIZE].to_csv(SECOND_BATCH_PATH, index=False)

    print(f"Created {FIRST_BATCH_PATH.name}: source rows 1-{BATCH_SIZE}.")
    print(
        f"Created {SECOND_BATCH_PATH.name}: "
        f"source rows {BATCH_SIZE + 1}-{2 * BATCH_SIZE}."
    )
    print("Both files preserve the source columns and exclude TARGET.")


if __name__ == "__main__":
    main()
