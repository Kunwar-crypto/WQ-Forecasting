# LSTM modelling

This folder contains the general-purpose LSTM forward-chain forecasting workflow for continuous water-quality variables.

The workflow supports targets such as:

- Dissolved oxygen
- Electrical conductivity
- Turbidity
- Total suspended solids

The model integrates:

- historical environmental predictor sequences;
- target-month environmental predictors;
- lagged water-quality values;
- cyclical month sine and cosine features;
- recursive multi-lead forecasting; and
- expanding-window model training.

The workflow produces forecasts using:

- reference or reanalysis predictors; and
- lead-specific seasonal forecast predictors.

Model performance is evaluated using:

- Kling–Gupta Efficiency
- Normalized Mean Absolute Error

The target variable, predictors, input paths, forecast leads, lag structure, training period, model hyperparameters, and output directory are specified in the accompanying JSON configuration file.

The original water-quality observations and processed modelling datasets are not included in this repository.
