# Feature selection

This folder contains the general-purpose Boruta feature-selection workflow for continuous water-quality modelling.

The script can be applied to water-quality targets such as dissolved oxygen, electrical conductivity, turbidity, and total suspended solids.

Users can specify their own:

- target variable;
- predictor variables;
- input file and worksheet;
- training-period cutoff;
- Random Forest settings; and
- output directory.

The workflow classifies predictors as confirmed, tentative, or rejected and saves feature-ranking tables, selected-feature lists, model settings, and ranking figures.

The original water-quality and predictor datasets are not included in this repository.
