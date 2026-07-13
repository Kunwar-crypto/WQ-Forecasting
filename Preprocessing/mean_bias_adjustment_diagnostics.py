"""
General-purpose Mean Bias Adjustment (MBA) and systematic-error diagnostics.

This script processes lead-specific Excel files containing seasonal-forecast
and reference/reanalysis variables. It:

1. Applies additive mean bias adjustment either by calendar month or globally.
2. Optionally clips physically non-negative variables at zero.
3. Calculates diagnostics before and after MBA:
   - Mean Bias Error (MBE)
   - Pearson correlation coefficient
   - Variance ratio
4. Saves:
   - adjusted lead-wise datasets;
   - monthly/global bias values;
   - a consolidated diagnostic table in CSV and Excel formats;
   - heatmaps of post-adjustment correlation and variance ratio.

The default column names reproduce the hydroclimatic variables used in the
associated study, but all paths, file patterns, variables, and column names
can be changed in the USER CONFIGURATION section.

Expected default input files
----------------------------
Lead 1.xlsx
Lead 2.xlsx
...
Lead 6.xlsx

Expected default columns
------------------------
Target Month
Rainfall (C3S)
Rainfall (ERA5)
Surface Runoff (C3S)
Surface Runoff (ERA5)
Temperature (C3S)
Temperature (ERA5)

Example
-------
python mean_bias_adjustment_diagnostics.py \
    --input-dir data/lead_files \
    --output-dir outputs/mba
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# The filename pattern must contain the placeholder {lead}.
DEFAULT_FILE_PATTERN = "Lead {lead}.xlsx"
DEFAULT_LEADS = tuple(range(1, 7))
DEFAULT_DATE_COLUMN = "Target Month"
DEFAULT_SHEET_NAME = 0

# "monthly" removes the mean forecast-minus-reference bias separately for each
# calendar month. "global" removes one overall mean bias per lead and variable.
DEFAULT_GROUPING = "monthly"

# Variable configuration:
# forecast_col  = seasonal-forecast column to adjust
# reference_col = reference/reanalysis column
# unit          = unit written to the diagnostic table
# nonnegative   = whether adjusted values should be clipped at zero
VARIABLES: Mapping[str, Mapping[str, object]] = {
    "Rainfall": {
        "forecast_col": "Rainfall (C3S)",
        "reference_col": "Rainfall (ERA5)",
        "unit": "mm/month",
        "nonnegative": True,
    },
    "Surface Runoff": {
        "forecast_col": "Surface Runoff (C3S)",
        "reference_col": "Surface Runoff (ERA5)",
        "unit": "mm/month",
        "nonnegative": True,
    },
    "Temperature": {
        "forecast_col": "Temperature (C3S)",
        "reference_col": "Temperature (ERA5)",
        "unit": "°C",
        "nonnegative": False,
    },
}

SAVE_ADJUSTED_LEAD_FILES = True
CREATE_HEATMAPS = True


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Read command-line options while retaining sensible defaults."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply mean bias adjustment to lead-specific forecast files and "
            "calculate systematic-error diagnostics."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data") / "lead_files",
        help="Directory containing the lead-specific Excel files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "mean_bias_adjustment",
        help="Directory in which results will be written.",
    )
    parser.add_argument(
        "--file-pattern",
        default=DEFAULT_FILE_PATTERN,
        help=(
            "Input filename pattern containing '{lead}', for example "
            "'Lead {lead}.xlsx'."
        ),
    )
    parser.add_argument(
        "--leads",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEADS),
        help="Forecast lead numbers to process. Default: 1 2 3 4 5 6.",
    )
    parser.add_argument(
        "--date-column",
        default=DEFAULT_DATE_COLUMN,
        help="Column containing the target or valid month.",
    )
    parser.add_argument(
        "--sheet-name",
        default=DEFAULT_SHEET_NAME,
        help=(
            "Excel worksheet name or zero-based index. The default value 0 "
            "reads the first worksheet."
        ),
    )
    parser.add_argument(
        "--grouping",
        choices=("monthly", "global"),
        default=DEFAULT_GROUPING,
        help="Apply calendar-month-specific or global additive bias correction.",
    )
    parser.add_argument(
        "--skip-adjusted-files",
        action="store_true",
        help="Do not save lead-wise datasets containing adjusted columns.",
    )
    parser.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Do not create diagnostic heatmaps.",
    )
    return parser.parse_args()


def normalize_sheet_name(value: object) -> object:
    """Convert a numeric worksheet argument supplied as text to an integer."""
    if isinstance(value, str) and re.fullmatch(r"\d+", value):
        return int(value)
    return value


# =============================================================================
# DATA PREPARATION
# =============================================================================

def ensure_month_column(
    dataframe: pd.DataFrame,
    date_column: str,
) -> pd.DataFrame:
    """Return a copy containing a valid integer calendar-month column."""
    work = dataframe.copy()

    if "month" in work.columns:
        work["month"] = pd.to_numeric(
            work["month"],
            errors="coerce",
        )
    else:
        if date_column not in work.columns:
            raise KeyError(
                f"Neither a 'month' column nor the date column "
                f"'{date_column}' was found."
            )

        work[date_column] = pd.to_datetime(
            work[date_column],
            errors="coerce",
        )
        work["month"] = work[date_column].dt.month

    invalid_months = work["month"].notna() & ~work["month"].between(1, 12)
    if invalid_months.any():
        raise ValueError(
            "The month column contains values outside the range 1-12."
        )

    return work


def validate_required_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    context: str,
) -> None:
    """Raise a clear error when one or more required columns are absent."""
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise KeyError(f"{context}: missing required columns {missing}")


def paired_values(
    predicted: pd.Series,
    observed: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite forecast-reference pairs as NumPy arrays."""
    paired = pd.concat(
        [
            pd.to_numeric(predicted, errors="coerce"),
            pd.to_numeric(observed, errors="coerce"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    return (
        paired.iloc[:, 0].to_numpy(dtype=float),
        paired.iloc[:, 1].to_numpy(dtype=float),
    )


# =============================================================================
# DIAGNOSTIC METRICS
# =============================================================================

def mean_bias_error(
    predicted: pd.Series,
    observed: pd.Series,
) -> float:
    """Calculate MBE = mean(predicted - observed)."""
    forecast, reference = paired_values(predicted, observed)
    if len(forecast) == 0:
        return np.nan
    return float(np.mean(forecast - reference))


def pearson_correlation(
    predicted: pd.Series,
    observed: pd.Series,
) -> float:
    """Calculate the Pearson correlation coefficient."""
    forecast, reference = paired_values(predicted, observed)

    if (
        len(forecast) < 2
        or np.isclose(np.std(forecast, ddof=1), 0)
        or np.isclose(np.std(reference, ddof=1), 0)
    ):
        return np.nan

    return float(np.corrcoef(forecast, reference)[0, 1])


def variance_ratio(
    predicted: pd.Series,
    observed: pd.Series,
) -> float:
    """Calculate sample variance(predicted) / sample variance(observed)."""
    forecast, reference = paired_values(predicted, observed)

    if len(forecast) < 2:
        return np.nan

    reference_variance = np.var(reference, ddof=1)
    if np.isclose(reference_variance, 0):
        return np.nan

    return float(np.var(forecast, ddof=1) / reference_variance)


# =============================================================================
# MEAN BIAS ADJUSTMENT
# =============================================================================

def mean_bias_adjustment(
    dataframe: pd.DataFrame,
    forecast_column: str,
    reference_column: str,
    grouping: str,
    nonnegative: bool,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Apply additive mean bias adjustment.

    Monthly adjustment:
        corrected_t = forecast_t - mean(forecast - reference | calendar month)

    Global adjustment:
        corrected_t = forecast_t - mean(forecast - reference)

    Returns
    -------
    corrected:
        Bias-adjusted forecast series.
    bias_table:
        Calendar-month or global bias values used in the adjustment.
    """
    work = dataframe.copy()

    forecast = pd.to_numeric(
        work[forecast_column],
        errors="coerce",
    )
    reference = pd.to_numeric(
        work[reference_column],
        errors="coerce",
    )
    raw_bias = forecast - reference

    if grouping == "monthly":
        if "month" not in work.columns:
            raise KeyError(
                "A month column is required for monthly bias adjustment."
            )

        bias_values = raw_bias.groupby(work["month"]).mean()
        corrected = forecast - work["month"].map(bias_values)

        bias_table = (
            bias_values.rename("Mean Bias")
            .rename_axis("Month")
            .reset_index()
        )
        bias_table["Grouping"] = "monthly"

    elif grouping == "global":
        global_bias = float(raw_bias.mean())
        corrected = forecast - global_bias

        bias_table = pd.DataFrame(
            {
                "Grouping": ["global"],
                "Month": [np.nan],
                "Mean Bias": [global_bias],
            }
        )

    else:
        raise ValueError("grouping must be either 'monthly' or 'global'")

    if nonnegative:
        corrected = corrected.clip(lower=0)

    return corrected, bias_table


# =============================================================================
# FORMATTING AND PLOTTING
# =============================================================================

def formatted_diagnostic_table(
    diagnosis: pd.DataFrame,
) -> pd.DataFrame:
    """Create a publication-friendly formatted copy of the diagnostic table."""
    table = diagnosis.copy()

    metric_columns = [
        "MBE before MBA",
        "MBE after MBA",
        "Pearson r before MBA",
        "Pearson r after MBA",
        "Variance ratio before MBA",
        "Variance ratio after MBA",
    ]

    for column in metric_columns:
        table[column] = table[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )

    return table


def annotate_heatmap(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    decimals: int = 2,
) -> None:
    """Write finite cell values over an imshow heatmap."""
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix.iloc[row_index, column_index]
            if pd.notna(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.{decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )


def create_post_adjustment_heatmaps(
    diagnosis: pd.DataFrame,
    output_directory: Path,
    variable_order: Sequence[str],
) -> Path:
    """Create the post-MBA correlation and variance-ratio heatmaps."""
    correlation_matrix = diagnosis.pivot(
        index="Variable",
        columns="Lead",
        values="Pearson r after MBA",
    ).reindex(variable_order)

    variance_matrix = diagnosis.pivot(
        index="Variable",
        columns="Lead",
        values="Variance ratio after MBA",
    ).reindex(variable_order)

    leads = correlation_matrix.columns.to_list()

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 4.5),
        constrained_layout=True,
    )

    correlation_image = axes[0].imshow(
        correlation_matrix.to_numpy(dtype=float),
        aspect="auto",
        vmin=-1,
        vmax=1,
    )
    axes[0].set_title("(a) Pearson correlation coefficient after MBA")
    axes[0].set_xlabel("Forecast lead (months)")
    axes[0].set_xticks(np.arange(len(leads)))
    axes[0].set_xticklabels(leads)
    axes[0].set_yticks(np.arange(len(variable_order)))
    axes[0].set_yticklabels(variable_order)
    annotate_heatmap(axes[0], correlation_matrix)
    colorbar_correlation = figure.colorbar(
        correlation_image,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )
    colorbar_correlation.set_label("Pearson r")

    variance_values = variance_matrix.to_numpy(dtype=float)
    finite_variance = variance_values[np.isfinite(variance_values)]

    if finite_variance.size == 0:
        distance_from_one = 0.1
    else:
        distance_from_one = float(
            np.max(np.abs(finite_variance - 1.0))
        )
        if np.isclose(distance_from_one, 0):
            distance_from_one = 0.1

    variance_image = axes[1].imshow(
        variance_values,
        aspect="auto",
        vmin=1.0 - distance_from_one,
        vmax=1.0 + distance_from_one,
    )
    axes[1].set_title("(b) Variance ratio after MBA")
    axes[1].set_xlabel("Forecast lead (months)")
    axes[1].set_xticks(np.arange(len(leads)))
    axes[1].set_xticklabels(leads)
    axes[1].set_yticks(np.arange(len(variable_order)))
    axes[1].set_yticklabels(variable_order)
    annotate_heatmap(axes[1], variance_matrix)
    colorbar_variance = figure.colorbar(
        variance_image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )
    colorbar_variance.set_label("Variance ratio")

    output_path = (
        output_directory
        / "heatmap_correlation_and_variance_ratio_after_mba.png"
    )
    figure.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(figure)

    return output_path


# =============================================================================
# MAIN PROCESSING WORKFLOW
# =============================================================================

def process_lead_files(
    input_directory: Path,
    output_directory: Path,
    file_pattern: str,
    leads: Sequence[int],
    date_column: str,
    sheet_name: object,
    grouping: str,
    save_adjusted_files: bool,
    create_heatmaps: bool,
) -> pd.DataFrame:
    """Process all requested lead files and save diagnostics."""
    if "{lead}" not in file_pattern:
        raise ValueError(
            "The input file pattern must contain the placeholder '{lead}'."
        )

    input_directory = input_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    adjusted_directory = output_directory / "adjusted_lead_files"
    bias_directory = output_directory / "bias_values"

    if save_adjusted_files:
        adjusted_directory.mkdir(parents=True, exist_ok=True)
    bias_directory.mkdir(parents=True, exist_ok=True)

    diagnostic_rows = []
    bias_tables = []

    for lead in sorted(set(leads)):
        input_path = input_directory / file_pattern.format(lead=lead)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        print(f"Processing lead {lead}: {input_path.name}")

        dataframe = pd.read_excel(
            input_path,
            sheet_name=sheet_name,
        )
        dataframe = ensure_month_column(
            dataframe,
            date_column=date_column,
        )

        for variable_name, configuration in VARIABLES.items():
            forecast_column = str(configuration["forecast_col"])
            reference_column = str(configuration["reference_col"])
            unit = str(configuration["unit"])
            nonnegative = bool(configuration["nonnegative"])

            validate_required_columns(
                dataframe,
                [forecast_column, reference_column],
                context=f"Lead {lead}, {variable_name}",
            )

            adjusted_column = f"{forecast_column} (MBA)"

            adjusted, bias_table = mean_bias_adjustment(
                dataframe=dataframe,
                forecast_column=forecast_column,
                reference_column=reference_column,
                grouping=grouping,
                nonnegative=nonnegative,
            )
            dataframe[adjusted_column] = adjusted

            bias_table.insert(0, "Lead", lead)
            bias_table.insert(1, "Variable", variable_name)
            bias_table.insert(2, "Unit", unit)
            bias_tables.append(bias_table)

            diagnostic_rows.append(
                {
                    "Lead": lead,
                    "Variable": variable_name,
                    "Unit": unit,
                    "N paired": len(
                        paired_values(
                            dataframe[forecast_column],
                            dataframe[reference_column],
                        )[0]
                    ),
                    "MBE before MBA": mean_bias_error(
                        dataframe[forecast_column],
                        dataframe[reference_column],
                    ),
                    "MBE after MBA": mean_bias_error(
                        dataframe[adjusted_column],
                        dataframe[reference_column],
                    ),
                    "Pearson r before MBA": pearson_correlation(
                        dataframe[forecast_column],
                        dataframe[reference_column],
                    ),
                    "Pearson r after MBA": pearson_correlation(
                        dataframe[adjusted_column],
                        dataframe[reference_column],
                    ),
                    "Variance ratio before MBA": variance_ratio(
                        dataframe[forecast_column],
                        dataframe[reference_column],
                    ),
                    "Variance ratio after MBA": variance_ratio(
                        dataframe[adjusted_column],
                        dataframe[reference_column],
                    ),
                }
            )

        if save_adjusted_files:
            output_name = f"lead_{lead}_mba_adjusted.xlsx"
            dataframe.to_excel(
                adjusted_directory / output_name,
                index=False,
            )

    diagnosis = (
        pd.DataFrame(diagnostic_rows)
        .sort_values(["Variable", "Lead"])
        .reset_index(drop=True)
    )

    all_bias_values = (
        pd.concat(bias_tables, ignore_index=True)
        .sort_values(["Variable", "Lead", "Month"])
        .reset_index(drop=True)
    )

    # Save machine-readable tables.
    diagnosis_csv = output_directory / "mba_systematic_error_diagnosis.csv"
    diagnosis_excel = output_directory / "mba_systematic_error_diagnosis.xlsx"
    bias_csv = bias_directory / "mba_bias_values.csv"
    bias_excel = bias_directory / "mba_bias_values.xlsx"

    diagnosis.to_csv(diagnosis_csv, index=False)
    all_bias_values.to_csv(bias_csv, index=False)

    with pd.ExcelWriter(diagnosis_excel) as writer:
        diagnosis.to_excel(
            writer,
            sheet_name="Diagnostics_raw",
            index=False,
        )
        formatted_diagnostic_table(diagnosis).to_excel(
            writer,
            sheet_name="Diagnostics_formatted",
            index=False,
        )

    all_bias_values.to_excel(bias_excel, index=False)

    print(f"Saved diagnostics: {diagnosis_csv}")
    print(f"Saved diagnostics: {diagnosis_excel}")
    print(f"Saved bias values: {bias_csv}")
    print(f"Saved bias values: {bias_excel}")

    if create_heatmaps:
        heatmap_path = create_post_adjustment_heatmaps(
            diagnosis=diagnosis,
            output_directory=output_directory,
            variable_order=list(VARIABLES.keys()),
        )
        print(f"Saved heatmap: {heatmap_path}")

    return diagnosis


def main() -> None:
    """Run the repository-ready MBA workflow."""
    arguments = parse_arguments()

    sheet_name = normalize_sheet_name(arguments.sheet_name)
    save_adjusted_files = (
        SAVE_ADJUSTED_LEAD_FILES
        and not arguments.skip_adjusted_files
    )
    create_heatmaps = (
        CREATE_HEATMAPS
        and not arguments.skip_heatmaps
    )

    print(f"Input directory: {arguments.input_dir.resolve()}")
    print(f"Output directory: {arguments.output_dir.resolve()}")
    print(f"File pattern: {arguments.file_pattern}")
    print(f"Leads: {arguments.leads}")
    print(f"Grouping: {arguments.grouping}")

    try:
        diagnosis = process_lead_files(
            input_directory=arguments.input_dir,
            output_directory=arguments.output_dir,
            file_pattern=arguments.file_pattern,
            leads=arguments.leads,
            date_column=arguments.date_column,
            sheet_name=sheet_name,
            grouping=arguments.grouping,
            save_adjusted_files=save_adjusted_files,
            create_heatmaps=create_heatmaps,
        )
    except Exception as error:
        print(f"Processing failed: {error}", file=sys.stderr)
        raise

    print("\nDiagnostic table preview:")
    print(diagnosis.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
