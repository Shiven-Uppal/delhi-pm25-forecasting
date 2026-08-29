# Time-Aware Multi-Station Forecasting of Hourly PM2.5 Across 31 Delhi Monitoring Stations
This repository contains the analysis code, derived results and figures for a study evaluating time-aware methods for forecasting hourly PM2.5 concentrations at 1-, 6- and 24-hour horizons across 31 Delhi monitoring stations.

## Study design
- Public fifteen-minute monitoring observations from January 2024 to December 2025
- Hourly aggregation and documented quality-control rules
- Chronological training, validation and future-year test periods
- Benchmark comparison with persistence, seasonal persistence, Ridge regression and histogram gradient boosting
- Robustness checks using blocked bootstrap inference, station-held-out transfer and grouped permutation importance

## Main result
The final PM10-free six-hour histogram-gradient-boosting model achieved an MAE of 32.05 micrograms per cubic metre on the 2025 test period, representing a 25.1% lower MAE than persistence.

## Repository contents
- Python scripts in the repository root: data audit, hourly aggregation, feature construction, model evaluation, robustness checks and figure generation
- `results/`: derived numerical outputs
- `figures/`: figures used in the report
- `docs/`: data-access and reproducibility notes

## Scope
This is a predictive forecasting study. It does not estimate a city-wide pollution field, identify emission sources, or demonstrate physical atmospheric transport or causal effects.

## Data access
The raw monitoring data are publicly archived and are not redistributed here. See `docs/data_access.md` for access and processing information.
