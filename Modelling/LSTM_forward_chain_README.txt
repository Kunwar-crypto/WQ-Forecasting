GENERAL-PURPOSE LSTM FORWARD-CHAIN FORECASTING
================================================

Files
-----
lstm_forward_chain.py
lstm_forward_chain_config.json
lstm_forward_chain_requirements.txt

Purpose
-------
The script trains a dual-input stacked LSTM for continuous water-quality
forecasting and compares:

1. Reference branch
   Target-month predictors from the reference/reanalysis dataset.

2. SeasonalForecast branch
   Lead-specific target-month predictors from seasonal forecast files.

It can be used for dissolved oxygen, electrical conductivity, turbidity,
total suspended solids, or another numeric water-quality target.

Only the following performance metrics are calculated:

- Kling-Gupta Efficiency (KGE)
- Normalized Mean Absolute Error (MAE%)

R-squared, raw MAE, RMSE, Monte Carlo dropout, and prediction intervals are
not included.

Installation
------------
pip install -r lstm_forward_chain_requirements.txt

Run
---
python lstm_forward_chain.py --config lstm_forward_chain_config.json

Input structure
---------------
Reference file:

data/ERA5_Monthly.xlsx

The default example expects:

Month_Start
Turbidity
Rainfall
Surface Runoff
Air Temperature
NDTI

Lead-specific files:

data/c3s_lead_files/
├── Rolling_Forecast_Lead_1.xlsx
├── Rolling_Forecast_Lead_2.xlsx
├── ...
└── Rolling_Forecast_Lead_6.xlsx

Each lead file must contain a target-month column and all configured
future_features.

The original ERA5, C3S, remote-sensing, and water-quality data are not
distributed with the repository.

Configuration
-------------
Edit lstm_forward_chain_config.json to change:

- target variable;
- historical predictors;
- future predictors;
- file names and folders;
- forecast leads;
- target lags;
- look-back length;
- issue period;
- model hyperparameters;
- output location.

Feature names containing spaces must exactly match the input files.

Remote-sensing climatology
--------------------------
The script does not generate remote-sensing climatology internally. For an
ex-ante experiment, any month-of-year remote-sensing climatology values should
already be present in the corresponding lead-specific forecast files under
the configured future-feature column names.

Retraining frequency
--------------------
The configuration supports:

monthly
    Retrain using data ending one month before every issue month.

yearly
    Train once per forecast year using data ending in December of the
    preceding year. The trained model is reused for all issue months within
    that year.

fixed
    Use the same fixed training end for all issue months. Set fixed_train_end
    in the configuration.

Target lags
-----------
The default lag_steps are [1, 2], representing the two most recent target
states relative to the forecast target month.

During training, post-issue lag values use observed training-period targets
as teacher forcing. During evaluation, those values are replaced recursively
with earlier predictions. The issue-month target is represented by the
configured nowcast policy.

Nowcast policy
--------------
persistence
    Uses the preceding observed month as the issue-month target estimate.

observed
    Uses the observed issue-month target. Use this only when it would genuinely
    be available at the forecast issue time.

Validation
----------
The final fraction of chronologically ordered training samples is reserved
for validation. Training is performed with shuffle=False. This avoids the
random train-validation mixing present in many default Keras workflows.

Metrics
-------
KGE is calculated using correlation, variability ratio, and mean-bias ratio.

MAE% is calculated as:

100 × mean(|prediction - observation|) / mean(|observation|)



Main outputs
------------
outputs/lstm_forward_chain/
├── <run>_<target>_forecasts.csv
├── <run>_<target>_metrics_all.csv
├── <run>_<target>_metrics_<year>.csv
├── <run>_<target>_metrics.xlsx
├── <run>_<target>_per_lead_predictions.xlsx
├── model_architecture.txt
├── configuration_used.json
└── training_histories/


