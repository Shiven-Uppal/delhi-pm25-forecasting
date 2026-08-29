# Derived results

This folder contains derived numerical outputs used in the final report. It does not contain raw monitoring records or individual-level prediction files.

| File | Contents |
|---|---|
| `pm25_2024_25_coverage_audit.csv` | Station-level audit of the public 2024-2025 archive. |
| `fair_benchmark_results_2025.csv` | Chronological benchmark results for baseline and machine-learning models. |
| `six_hour_pm10_validation_ablation.csv` | Validation-only comparison used to select the PM10-free six-hour specification. |
| `final_pm10_free_six_hour_test_metrics.csv` | Final PM10-free six-hour model performance on the 2025 test period. |
| `final_pm10_free_six_hour_inference.csv` | Blocked-bootstrap confidence interval and Diebold-Mariano test against persistence. |
| `final_pm10_free_station_heldout_six_hour_results.csv` | Transfer performance for stations held out from training. |
| `final_pm10_free_grouped_permutation_importance.csv` | Grouped permutation-importance results for the final model. |
