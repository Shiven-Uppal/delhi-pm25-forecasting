# Data access and reproducibility

## Source archive

The analysis used publicly archived fifteen-minute Delhi monitoring observations from **1 January 2024 to 31 December 2025**. The report cites the OpenCity Delhi air-quality archive:

<https://data.opencity.in/dataset/delhi-hourly-air-quality-reports>

The raw records are not redistributed in this repository. Download the same 2024-2025 archive before attempting an exact rerun, and retain the files unchanged in a local raw-data directory.

## Expected content

The processing scripts expect station-level CSV files containing, at minimum:

- `Timestamp`
- `Station ID`
- `Station Name`
- `PM2.5 (µg/m³)`

The archived files may additionally contain co-pollutants and local meteorological measurements. These are used when available. The hourly aggregation script currently identifies raw files whose names contain `15_minute`.

## Processing rules

- Timestamps are parsed as UTC and assigned to their enclosing clock hour.
- PM2.5 is retained for an hour only when at least three quarter-hour readings are present.
- Other numeric variables are aggregated as hourly means, while their quarter-hour observation counts are retained.
- Valid high-pollution events are not removed as outliers.
- Predictor imputation and scaling are fitted on the training partition only in the modelling scripts.

## Core-station cohort

The 31-station core cohort was defined before model fitting by applying the stated 10% hourly-PM2.5 missingness threshold across the complete 2024-2025 study archive. This uses 2025 **availability** only to define the fixed study cohort; no 2025 forecast errors, target outcomes, hyperparameters, or model-selection results were used to choose the model. Readers should distinguish this cohort-definition step from the chronologically untouched 2025 performance evaluation.

## Exact reproducibility note

Public archives can be amended after publication. For an exact reproduction of the reported figures, retain a local copy of the 2024-2025 source files used for the study and record its retrieval date and file checksums. The committed derived results in [`results/`](../results/README.md) are the numerical outputs reported in the paper.
