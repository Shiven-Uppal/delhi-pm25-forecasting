# Reproduction guide

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The public raw Delhi monitoring records are not included in this repository. See [data_access.md](data_access.md) for the data description.

To rerun the workflow, obtain the same 2024-2025 public archive and update the input and output folder paths at the top of the Python scripts to match your computer.

## Analysis order

Run the scripts in this order:

1. `audit_pm25_archive.py`
2. `build_hourly_pm25_panel.py`
3. `build_forecasting_features.py`
4. `run_fair_benchmarks.py`
5. `run_pm10_validation_check.py`
6. `create_final_2025_evaluation.py`
7. `final_pm10_free_six_hour_evaluation.py`
8. `test_six_hour_improvement.py`
9. `run_station_heldout_test.py`
10. `grouped_permutation_importance.py`
11. `create_final_paper_figures.py`
12. `create_residual_diagnostics_figure.py`

## Included outputs

The numerical results reported in the paper are in `results/`. Report figures are in `figures/`.

Exact reruns require access to the same public raw-data archive and local path settings.
