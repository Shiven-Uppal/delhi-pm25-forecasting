# Time-Aware Multi-Station Forecasting of Hourly PM2.5 Across 31 Delhi Monitoring Stations

This repository contains the analysis code, derived results, and figures for a study evaluating time-aware methods for forecasting hourly PM2.5 concentrations at 1-, 6-, and 24-hour horizons across 31 Delhi monitoring stations.

## Main finding

On the pre-specified 2025 test period, the final PM10-free six-hour histogram-gradient-boosting model achieved an MAE of **32.05 µg/m³**, compared with **42.76 µg/m³** for persistence: a **25.1% reduction in MAE**. This result is specific to the six-hour horizon; persistence was the validation-selected model at one and 24 hours.

## Study design

- Public fifteen-minute monitoring observations from January 2024 to December 2025
- Quality-controlled hourly aggregation
- Chronological training, validation, and future-year testing
- Comparison with persistence, seasonal persistence, Ridge regression, and histogram gradient boosting
- Robustness checks using blocked-bootstrap inference, a Diebold-Mariano test, station-held-out transfer, and grouped permutation importance

## Reproduce the analysis

See the step-by-step [reproduction guide](docs/reproduction.md) and [data-access note](docs/data_access.md). The raw monitoring records are public but are not redistributed here. Exact reruns require the same 2024-2025 archive and the original data schema.

```bash
pip install -r requirements.txt
```

Then set local input/output paths in the scripts as described in the reproduction guide and run the workflow in its documented order.

## Repository contents

- Root Python scripts: data audit, hourly aggregation, feature construction, model evaluation, robustness checks, and figure generation
- [`results/`](results/README.md): numerical outputs reported in the paper
- [`figures/`](figures/): report figures
- [`docs/`](docs/): data-access and reproduction notes
- [`CITATION.cff`](CITATION.cff): citation metadata

## Scope and interpretation

This is a predictive forecasting study. It does not estimate a city-wide pollution field, identify emission sources, demonstrate physical atmospheric transport, or support causal claims. The paper’s uncertainty analysis concerns the average comparative performance of the six-hour model versus persistence; it is not a per-forecast prediction interval.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also link to this repository when using the code or derived outputs.
