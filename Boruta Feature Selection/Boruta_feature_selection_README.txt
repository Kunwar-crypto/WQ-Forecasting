GENERAL-PURPOSE BORUTA FEATURE SELECTION
=======================================

Main file
---------
boruta_feature_selection.py

Purpose
-------
This script performs Boruta feature selection for any continuous
water-quality target, including:

- Dissolved oxygen
- Electrical conductivity
- Turbidity
- Total suspended solids
- Any other numeric target supplied by the user

Installation
------------
pip install -r boruta_requirements.txt

Basic example
-------------
python boruta_feature_selection.py \
    --input-file data/model_inputs.xlsx \
    --sheet-name Sheet1 \
    --target "Conductivity" \
    --predictors B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 NDTI NDVI "B4/B8" "B4/B2" \
    --output-dir outputs/boruta_conductivity

Leakage-aware example
---------------------
To restrict feature screening to observations available before forecasting:

python boruta_feature_selection.py \
    --input-file data/model_inputs.xlsx \
    --sheet-name Sheet1 \
    --target "Conductivity" \
    --predictors B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 NDTI NDVI "B4/B8" "B4/B2" \
    --date-column "Target Month" \
    --training-end 2021-12-31 \
    --output-dir outputs/boruta_conductivity

Automatic predictor selection
-----------------------------
If --predictors is omitted, the script uses all numeric columns except:

- the target;
- the optional date column;
- columns supplied through --exclude-columns.

Example:

python boruta_feature_selection.py \
    --input-file data/model_inputs.csv \
    --target TSS \
    --exclude-columns Station Lead \
    --output-dir outputs/boruta_tss

Skip Random Forest tuning
-------------------------
python boruta_feature_selection.py \
    --input-file data/model_inputs.xlsx \
    --target DO \
    --no-tuning

Outputs
-------
The output directory contains:

- boruta_feature_ranking.csv
- boruta_feature_ranking.xlsx
- confirmed_features.txt
- tentative_features.txt
- rejected_features.txt
- boruta_run_metadata.json
- boruta_feature_ranks_<target>.png

Important methodological notes
------------------------------
1. Random Forest and Boruta do not require predictor standardization, so
   StandardScaler is intentionally omitted.

2. For strict forward-chain forecasting, feature selection should ideally use
   only data available before the forecast period. The optional
   --training-end argument supports this workflow.

3. The script does not include or redistribute any original water-quality
   observations.
