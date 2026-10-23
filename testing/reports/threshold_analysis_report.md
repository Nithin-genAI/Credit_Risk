# Threshold Analysis — Existing Saved Baseline

> **Diagnostic caveat:** Neither subset is an independent final test.
> The original validation partition influenced early stopping and
> model iteration selection.

## Method

- Reused the saved XGBoost model and preprocessor; neither was fitted,
  tuned, overwritten, or regenerated.
- Reused complete benchmark feature rows and existing diagnostic
  partition membership. `TARGET` was kept outside model inputs and used
  only for metrics.
- Compared only the requested thresholds on `diagnostic_validation`.
- Selected a candidate by the highest F1 among the requested values;
  ties favor the higher threshold. This is a transparent technical
  heuristic, **not** a business objective or an optimal policy.
- Evaluated that candidate once on `secondary_holdout_diagnostic` as a
  provisional consistency check, not to select or revise the candidate.

- Diagnostic validation records: 30,751
- Secondary holdout diagnostic records: 30,752
- Diagnostic target counts: {'0': 28269, '1': 2482}
- Positive class: TARGET=1 (payment difficulty/default)
- Selected candidate threshold: **0.15**
- Candidate selection F1 (diagnostic validation): 0.318967

## Threshold comparison — diagnostic validation

| Threshold | Precision | Recall | F1 | TP | FP | TN | FN | Predicted positive |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.133642 | 0.838034 | 0.230522 | 2,080 | 13,484 | 14,785 | 402 | 15,564 |
| 0.10 | 0.195663 | 0.610798 | 0.296383 | 1,516 | 6,232 | 22,037 | 966 | 7,748 |
| 0.15 **candidate** | 0.249263 | 0.442788 | 0.318967 | 1,099 | 3,310 | 24,959 | 1,383 | 4,409 |
| 0.20 | 0.298535 | 0.320306 | 0.309038 | 795 | 1,868 | 26,401 | 1,687 | 2,663 |
| 0.30 | 0.398810 | 0.161966 | 0.230372 | 402 | 606 | 27,663 | 2,080 | 1,008 |
| 0.50 | 0.537190 | 0.026189 | 0.049942 | 65 | 56 | 28,213 | 2,417 | 121 |

The curve and marked candidate cutoffs are in `precision_recall_curve.svg`; underlying curve coordinates are in `precision_recall_curve.csv`.

## Provisional consistency check on secondary holdout diagnostic

At the diagnostic-selected threshold **0.15**: precision **0.255921**, recall **0.448248**, F1 **0.325820**; TP=1,113, FP=3,236, TN=25,033, FN=1,370.

This holdout result did not influence candidate selection. It is
still not independent: it is part of the original validation
partition that influenced early stopping.

## Interpreting the trade-off

Lowering the threshold labels more applicants as positive. This
typically raises recall by catching more actual TARGET=1 cases,
while also raising false positives and potentially lowering
precision. Higher thresholds usually make fewer positive calls;
precision may increase while recall falls. Select a deployment
threshold only after defining the relative costs of missed defaults
and false alarms and validating that objective on appropriate
independent data. No threshold here is declared optimal.

## Checks

- source_dataset_matches_existing_evaluation_hash: **PASS**
- partition_membership_matches_existing_evaluation: **PASS**
- partition_labels_and_source_ids_match_dataset: **PASS**
- TARGET_excluded_from_model_inputs: **PASS**
- saved_preprocessor_schema_matches_metadata: **PASS**
- transformed_feature_width_matches_model: **PASS**
- complete_benchmark_rows_used: **PASS**
- probability_and_label_alignment_verified: **PASS**
- diagnostic_only_threshold_candidate_selection: **PASS**
- model_artifact_unchanged: **PASS**
- preprocessor_artifact_unchanged: **PASS**

## Limitations

- Diagnostic results are not independent final-test estimates.
- The threshold-selection subset itself was part of the data used
  for early-stopping selection, so its metrics may be optimistic.
- The second subset is only a provisional consistency check.
- The fixed 0.50 cutoff is retained as one comparator, not a
  lending-policy recommendation.
