MEAN BIAS ADJUSTMENT AND SYSTEMATIC-ERROR DIAGNOSTICS
====================================================

Main file
---------
mean_bias_adjustment_diagnostics.py

Purpose
-------
This repository-ready script applies additive mean bias adjustment to
lead-specific C3S variables using corresponding ERA5 variables as the
reference. It also calculates:

- Mean Bias Error before and after adjustment
- Pearson correlation before and after adjustment
- Variance ratio before and after adjustment

Default input structure
-----------------------
Place the following files in:

data/lead_files/

Expected default filenames:

Lead 1.xlsx
Lead 2.xlsx
Lead 3.xlsx
Lead 4.xlsx
Lead 5.xlsx
Lead 6.xlsx

Expected default columns:

Target Month
Rainfall (C3S)
Rainfall (ERA5)
Surface Runoff (C3S)
Surface Runoff (ERA5)
Temperature (C3S)
Temperature (ERA5)

The input spreadsheets themselves are not included in the public repository.

Installation
------------
pip install numpy pandas matplotlib openpyxl

Run
---
python mean_bias_adjustment_diagnostics.py

Or specify custom directories:

python mean_bias_adjustment_diagnostics.py \
    --input-dir data/lead_files \
    --output-dir outputs/mean_bias_adjustment

Custom filename pattern
-----------------------
The file pattern must include {lead}. Example:

python mean_bias_adjustment_diagnostics.py \
    --file-pattern "Forecast_Lead_{lead}.xlsx"

Global rather than monthly adjustment
-------------------------------------
python mean_bias_adjustment_diagnostics.py --grouping global

Outputs
-------
outputs/mean_bias_adjustment/
├── mba_systematic_error_diagnosis.csv
├── mba_systematic_error_diagnosis.xlsx
├── heatmap_correlation_and_variance_ratio_after_mba.png
├── adjusted_lead_files/
│   ├── lead_1_mba_adjusted.xlsx
│   └── ...
└── bias_values/
    ├── mba_bias_values.csv
    └── mba_bias_values.xlsx

Method
------
For monthly adjustment:

corrected forecast =
raw forecast - mean(raw forecast - reference | calendar month)

For global adjustment:

corrected forecast =
raw forecast - mean(raw forecast - reference)

Rainfall and surface runoff are clipped at zero after adjustment by default.
This can be changed in the VARIABLES configuration.

Customization
-------------
Edit the VARIABLES dictionary near the beginning of the Python script to
change:

- variable names;
- forecast and reference column names;
- units;
- non-negative clipping.

Reproducibility note
--------------------
The script contains the complete bias-adjustment and diagnostic workflow but
does not redistribute the original ERA5, C3S, or water-quality datasets.
