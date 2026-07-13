# Seasonal River Water-Quality Forecasting

This repository contains the data-acquisition, preprocessing, feature-selection, and long short-term memory modelling codes developed for seasonal river water-quality forecasting.

The workflow integrates Sentinel-2 remote sensing, ERA5-Land reanalysis, Copernicus Climate Change Service seasonal forecasts, mean bias adjustment, Boruta feature selection, and forward-chain LSTM modelling.

The scripts can be adapted for continuous water-quality parameters such as:

- Dissolved oxygen
- Electrical conductivity
- Turbidity
- Total suspended solids

## Repository structure

```text
seasonal-water-quality-forecasting/
│
├── data_acquisition/
│   ├── GEE_Sentinel2_Surface_Reflectance_QA60_SCL.txt
│   ├── download_era5_land_monthly.py
│   └── download_c3s_system51_monthly.py
│
├── preprocessing/
│   └── mean_bias_adjustment_diagnostics.py
│
├── feature_selection/
│   └── boruta_feature_selection.py
│
├── modelling/
│   ├── lstm_forward_chain.py
│   └── lstm_forward_chain_config.json
│
├── requirements/
│   ├── MBA_requirements.txt
│   ├── boruta_requirements.txt
│   └── lstm_forward_chain_requirements.txt
│
└── README.md
```

## Workflow overview

The repository provides scripts for the following stages.

### 1. Sentinel-2 data acquisition

The Google Earth Engine script retrieves Sentinel-2 Level-2A surface-reflectance data.

Cloud-affected and invalid pixels are screened using:

- QA60 opaque-cloud and cirrus flags
- Scene Classification Layer cloud classes
- Cloud-shadow class
- Snow and ice class
- No-data and defective-pixel classes

The script also supports:

- surface-reflectance time-series extraction;
- region-mean reflectance export;
- pixel-level reflectance export; and
- cloud-masked GeoTIFF export.

The script is provided as a text file and should be copied into the Google Earth Engine Code Editor.

### 2. ERA5-Land data acquisition

The ERA5-Land script downloads monthly averaged data from the Copernicus Climate Data Store.

The default variables are:

- 2 m temperature
- Surface runoff
- Total precipitation

The example request covers January 2019 to December 2023.

### 3. C3S seasonal forecast acquisition

The C3S script downloads ECMWF System 51 seasonal monthly single-level forecasts.

The default configuration includes:

- Ensemble-mean forecasts
- 2 m temperature
- Mean surface runoff rate
- Total precipitation
- Forecast leads of 1–6 months
- Initialization years from 2018 to 2023

The script downloads one NetCDF file for each initialization month, allowing interrupted downloads to be resumed.

### 4. Mean bias adjustment

The preprocessing script applies additive mean bias adjustment to seasonal forecast variables using corresponding ERA5 variables as the reference.

The adjustment can be performed:

- separately for each calendar month; or
- using one global mean bias.

Rainfall and surface runoff values can be constrained to remain non-negative after adjustment.

The following systematic-error diagnostics are calculated before and after adjustment:

- Mean Bias Error
- Pearson correlation coefficient
- Variance ratio

### 5. Boruta feature selection

The Boruta script identifies informative predictors for a user-specified water-quality target.

Users can provide:

- any continuous water-quality target;
- custom spectral bands and indices;
- custom environmental predictors;
- Excel or CSV input data; and
- an optional training-period cutoff.

Predictors are classified as:

- Confirmed
- Tentative
- Rejected

Feature-selection tables, selected-feature lists, model settings, and ranking figures are saved automatically.

### 6. LSTM forward-chain forecasting

The modelling script implements a configurable stacked LSTM framework for multi-lead water-quality forecasting.

The workflow supports:

- historical predictor sequences;
- target-month environmental predictors;
- lagged water-quality values;
- month sine and cosine terms;
- forecast leads of 1–6 months;
- recursive prediction for longer leads;
- expanding-window training;
- reference-driven forecasts; and
- seasonal-forecast-driven forecasts.

The default stacked LSTM architecture contains:

- first LSTM layer with 32 units;
- second LSTM layer with 16 units;
- dense layers with ReLU activation;
- dropout regularization;
- Adam optimization; and
- Huber loss.

The model settings, target variable, predictors, forecast period, lags, look-back window, and output folders can be modified in:

```text
modelling/lstm_forward_chain_config.json
```

## Evaluation metrics

The LSTM workflow reports only the following performance metrics:

- Kling–Gupta Efficiency
- Normalized Mean Absolute Error

Normalized mean absolute error is calculated as:

```text
MAE (%) = 100 × mean(|Prediction − Observation|) / mean(|Observation|)
```

The workflow does not calculate R-squared, RMSE, or Monte Carlo dropout uncertainty intervals.

## Input data

The original water-quality observations and processed modelling datasets are not included in this repository because of data-access restrictions.

Users must provide their own input files with column names corresponding to those specified in the scripts or configuration file.

The repository does not contain:

- original water-quality observations;
- processed ERA5 or C3S datasets;
- downloaded Sentinel-2 imagery;
- personal file paths;
- Google Earth Engine credentials; or
- Copernicus Climate Data Store credentials.

## Software requirements

The main Python packages used in the repository are:

- NumPy
- pandas
- scikit-learn
- TensorFlow
- Matplotlib
- Boruta
- openpyxl
- cdsapi

Package-specific requirement files are provided in the `requirements` directory.

## Installation

Clone or download the repository and install the required packages for the relevant workflow.

For mean bias adjustment:

```bash
pip install -r requirements/MBA_requirements.txt
```

For Boruta feature selection:

```bash
pip install -r requirements/boruta_requirements.txt
```

For LSTM forecasting:

```bash
pip install -r requirements/lstm_forward_chain_requirements.txt
```

The CDS API package can be installed using:

```bash
pip install cdsapi
```

## Running the scripts

### ERA5-Land download

```bash
python data_acquisition/download_era5_land_monthly.py
```

### C3S seasonal forecast download

```bash
python data_acquisition/download_c3s_system51_monthly.py
```

### Mean bias adjustment

```bash
python preprocessing/mean_bias_adjustment_diagnostics.py \
    --input-dir data/lead_files \
    --output-dir outputs/mean_bias_adjustment
```

### Boruta feature selection

Example for electrical conductivity:

```bash
python feature_selection/boruta_feature_selection.py \
    --input-file data/model_inputs.xlsx \
    --sheet-name Sheet1 \
    --target "Conductivity" \
    --predictors B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 NDTI NDVI "B4/B8" "B4/B2" \
    --output-dir outputs/boruta_conductivity
```

### LSTM forward-chain forecasting

Before running the model, edit:

```text
modelling/lstm_forward_chain_config.json
```

Then run:

```bash
python modelling/lstm_forward_chain.py \
    --config modelling/lstm_forward_chain_config.json
```

## Recommended execution order

```text
1. Retrieve Sentinel-2 surface-reflectance data
2. Download ERA5-Land data
3. Download C3S seasonal forecast data
4. Prepare and align the input datasets
5. Apply mean bias adjustment
6. Perform Boruta feature selection
7. Configure the LSTM model
8. Run forward-chain forecasting
9. Evaluate forecasts using KGE and MAE%
```

## Reproducibility

The scripts include configurable settings for:

- random seed;
- target variable;
- predictor variables;
- forecast leads;
- lag structure;
- look-back period;
- training and forecasting periods;
- model hyperparameters;
- input paths; and
- output paths.

Users should modify the configuration according to their own datasets and forecasting design.

## Code availability

This repository provides the codes used for:

- Sentinel-2 data retrieval and cloud masking;
- ERA5-Land data acquisition;
- C3S seasonal forecast acquisition;
- mean bias adjustment;
- systematic-error diagnosis;
- Boruta feature selection; and
- LSTM forward-chain water-quality forecasting.

