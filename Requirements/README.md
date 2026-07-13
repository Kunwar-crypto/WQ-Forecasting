# Software requirements

This folder contains package requirement files for the main Python workflows.

- `MBA_requirements.txt` contains packages required for mean bias adjustment and systematic-error diagnosis.
- `boruta_requirements.txt` contains packages required for Boruta feature selection.
- `lstm_forward_chain_requirements.txt` contains packages required for LSTM forward-chain forecasting.

The Copernicus Climate Data Store acquisition scripts additionally require the `cdsapi` package:

```bash
pip install cdsapi
