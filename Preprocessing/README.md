# Preprocessing

This folder contains the general-purpose mean bias adjustment and systematic-error diagnostic workflow.

The script applies additive mean bias adjustment to seasonal forecast variables using corresponding reference or reanalysis variables.

It calculates:

- Mean Bias Error before and after adjustment
- Pearson correlation coefficient before and after adjustment
- Variance ratio before and after adjustment

The workflow supports monthly or global bias adjustment and can process multiple forecast-lead files automatically.

The original input datasets are not included in this repository.
