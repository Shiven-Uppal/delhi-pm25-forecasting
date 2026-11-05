# Reproduction guide

## Setup

Use Python 3.10 or later, then install the declared package requirements:

```bash
python -m pip install -r requirements.txt
```

Download the 2024-2025 public archive described in [data_access.md](data_access.md). Raw observations are intentionally not committed to this repository.

Before running any scripts, update their local input and output paths to match your computer. In particular, `build_hourly_pm25_panel.py` expects the raw input directory and writes an hourly-panel directory; later scripts expect the resulting feature file in the working directory.

## Workflow

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

## Outputs

The committed outputs under [`results/`](../results/README.md) and [`figures/`](../figures/) correspond to the paper. Fresh runs write several CSV and PNG outputs to the working directory; compare them with the committed outputs, then move them into an output directory if desired.

## Environment note

`requirements.txt` specifies compatible minimum versions. For a byte-for-byte historical reproduction, use a frozen environment exported from the original run as well as the same source archive.
