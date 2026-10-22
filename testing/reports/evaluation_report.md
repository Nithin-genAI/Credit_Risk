# Saved Baseline Model: Secondary Diagnostic Evaluation

> **Interpretation warning:** These results are secondary diagnostics,
> not an independent final test estimate. The original validation
> partition influenced early stopping and model-iteration selection.

## Evaluation setup

- Source dataset: `data\application_train.csv`
- Source SHA-256: `52e96b895b1112e1c853f670e58372719c8441c5ed1c57ac2f7fad559d784f5f`
- Dataset rows: 307,511
- Original notebook split: 246,008 training / 61,503 validation
- Original split settings: {'test_size': 0.2, 'random_state': 42, 'stratify': 'TARGET'}
- Diagnostic partition split: {'validation_fraction': 0.5, 'secondary_holdout_fraction': 0.5, 'random_state': 42, 'stratify': 'TARGET'}
- Classification threshold: 0.50
- Transformed feature count: 254
- Source IDs and `TARGET` are excluded from model features.

## Partition membership and labels

| Partition | Rows | TARGET=0 | TARGET=1 | Source-row-index SHA-256 |
|---|---:|---:|---:|---|
| diagnostic_validation | 30,751 | 28,269 | 2,482 | `24ca5c16d85e6d7323f07a0abae45d9d7c5f80d45b354520a8ace149dfe64b82` |
| secondary_holdout_diagnostic | 30,752 | 28,269 | 2,483 | `11e0649dcf600039bb63dbdd584b9201a9c2a2b038dc0a520b1bd0b94d823f84` |

Every original validation row and its target label are listed in `validation_partition_membership.csv`.

## Metrics

| Partition | ROC-AUC | Average Precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| diagnostic_validation | 0.767923 | 0.259617 | 0.537190 | 0.026189 | 0.049942 |
| secondary_holdout_diagnostic | 0.772034 | 0.261199 | 0.553719 | 0.026983 | 0.051459 |

### Confusion matrices at threshold 0.50

**diagnostic_validation** — labels `[0, 1]`, rows are actual and columns are predicted:

```text
[[28213, 56], [2417, 65]]
```

**secondary_holdout_diagnostic** — labels `[0, 1]`, rows are actual and columns are predicted:

```text
[[28215, 54], [2416, 67]]
```

## Checks

- reconstructed_split_matches_notebook_row_counts: **PASS**
- training_and_validation_partitions_are_disjoint: **PASS**
- diagnostic_subsets_are_disjoint_and_cover_validation: **PASS**
- TARGET_excluded_from_model_features: **PASS**
- SK_ID_CURR_excluded_from_model_features: **PASS**
- saved_preprocessor_schema_matches_feature_metadata: **PASS**
- transformed_feature_order_and_width_match_saved_model: **PASS**
- prediction_and_label_rows_are_aligned: **PASS**
- complete_benchmark_features_used_without_all_missing_fallback: **PASS**

## Method implemented by `evaluate_saved_baseline.py`

```text
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
```

## Limitations

- This does not retrain or tune the saved model.
- The diagnostic subsets are drawn from the validation partition
  already used for early stopping; splitting it afterward does not
  make either subset independent of that selection.
- Metrics use the fixed 0.50 probability cutoff for class labels;
  ROC-AUC and Average Precision evaluate ranking across thresholds.
- These benchmark results do not establish production readiness or
  real-world lending suitability.
