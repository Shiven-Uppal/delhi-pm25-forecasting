# Data access and reproducibility

## Raw monitoring data
The analysis used publicly archived fifteen-minute air-quality observations from Delhi monitoring stations between 1 January 2024 and 31 December 2025.
The raw records are not redistributed in this repository. 
They are public-source records and may be large. 
This repository contains the code, derived outputs and figures needed to inspect the analysis workflow.

## Variables used
The raw archive included fifteen-minute PM2.5 measurements, which were aggregated to hourly values for forecasting, together with available pollutant concentrations, meteorological variables, timestamps and station identifiers.

## Processing workflow
1. Obtain the public fifteen-minute station records for 2024-2025.
2. Place the downloaded files in a local raw-data directory.
3. Update the input and output path variables at the top of each script to match local file locations.
4. Run the scripts in the order described in the repository README.

## Data handling
Only derived, non-raw outputs are included in `results/`. 
The analysis did not delete valid high-pollution observations as outliers. 
Hourly PM2.5 was retained only when the required underlying fifteen-minute observations were available, as documented in the analysis code and report.

## Scope
The repository supports reproducibility of a predictive forecasting analysis. 
It does not provide a city-wide pollution map, emissions inventory, physical dispersion model or causal attribution analysis.
