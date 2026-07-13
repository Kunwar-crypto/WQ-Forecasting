"""
General-purpose Boruta feature-selection workflow for water-quality modelling.

This script identifies informative predictors for any user-specified
water-quality target (for example, dissolved oxygen, electrical conductivity,
turbidity, or total suspended solids) using Boruta with a Random Forest
regressor.

Main capabilities
-----------------
1. Reads Excel or CSV input data.
2. Allows any target variable and predictor set.
3. Optionally restricts feature selection to an initial training period to
   avoid using future observations during predictor screening.
4. Optionally tunes the Random Forest estimator using RandomizedSearchCV.
5. Classifies predictors as Confirmed, Tentative, or Rejected.
6. Saves:
   - feature-ranking tables;
   - confirmed/tentative/rejected feature lists;
   - tuned Random Forest parameters;
   - a feature-ranking figure.

Important methodological note
-----------------------------
Standardization is not required for Random Forest or Boruta because tree-based
models are insensitive to linear rescaling of predictors. Therefore, scaling
has been omitted from this general workflow.

Example
-------
python boruta_feature_selection.py \
    --input-file data/model_inputs.xlsx \
    --sheet-name Sheet1 \
    --target "Electrical Conductivity" \
    --predictors B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 NDTI NDVI B4_B8 B4_B2 \
    --date-column Date \
    --training-end 2021-12-31 \
    --output-dir outputs/boruta_ec
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_RS_PREDICTORS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
    "NDTI",
    "NDVI",
    "B4/B8",
    "B4/B2",
]

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ITER_SEARCH = 25
DEFAULT_CV_FOLDS = 3
DEFAULT_BORUTA_MAX_ITER = 100
DEFAULT_BORUTA_PERCENTILE = 100
DEFAULT_ALPHA = 0.05


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Boruta feature selection for a user-specified water-quality "
            "target using a Random Forest regressor."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input Excel (.xlsx/.xls) or CSV file.",
    )
    parser.add_argument(
        "--sheet-name",
        default=0,
        help=(
            "Excel worksheet name or zero-based index. Ignored for CSV files. "
            "Default: first worksheet."
        ),
    )
    parser.add_argument(
        "--target",
        required=True,
        help=(
            "Name of the water-quality target column, for example "
            "'DO', 'EC', 'Turbidity', or 'TSS'."
        ),
    )
    parser.add_argument(
        "--predictors",
        nargs="*",
        default=None,
        help=(
            "Predictor columns. If omitted, all numeric columns except the "
            "target, date column, and excluded columns are used."
        ),
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=[],
        help="Additional columns to exclude from automatic predictor selection.",
    )
    parser.add_argument(
        "--date-column",
        default=None,
        help=(
            "Optional date column. Required when --training-end is supplied."
        ),
    )
    parser.add_argument(
        "--training-end",
        default=None,
        help=(
            "Optional final date used for feature screening, for example "
            "'2021-12-31'. Rows after this date are excluded."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "boruta_feature_selection",
        help="Directory for all outputs.",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip RandomizedSearchCV and use the default Random Forest settings.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Cross-validation folds used during Random Forest tuning.",
    )
    parser.add_argument(
        "--n-iter-search",
        type=int,
        default=DEFAULT_N_ITER_SEARCH,
        help="Number of RandomizedSearchCV parameter combinations.",
    )
    parser.add_argument(
        "--boruta-max-iter",
        type=int,
        default=DEFAULT_BORUTA_MAX_ITER,
        help="Maximum Boruta iterations.",
    )
    parser.add_argument(
        "--boruta-percentile",
        type=int,
        default=DEFAULT_BORUTA_PERCENTILE,
        help=(
            "Percentile of shadow-feature importance used as the Boruta "
            "threshold. Default: 100."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Boruta significance level.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by scikit-learn. Default: all processors.",
    )
    return parser.parse_args()


def normalize_sheet_name(value: object) -> object:
    """Convert numeric text such as '0' to an integer worksheet index."""
    if isinstance(value, str) and re.fullmatch(r"\d+", value):
        return int(value)
    return value


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_table(input_file: Path, sheet_name: object) -> pd.DataFrame:
    """Read an Excel or CSV input table."""
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    suffix = input_file.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_file, sheet_name=sheet_name)

    if suffix == ".csv":
        return pd.read_csv(input_file)

    raise ValueError(
        "Unsupported input format. Use .xlsx, .xls, or .csv."
    )


def restrict_to_training_period(
    dataframe: pd.DataFrame,
    date_column: Optional[str],
    training_end: Optional[str],
) -> pd.DataFrame:
    """
    Restrict predictor screening to observations available by training_end.

    This option is useful for forward-chain or operational forecasting studies
    because it prevents future observations from influencing feature selection.
    """
    if training_end is None:
        return dataframe.copy()

    if date_column is None:
        raise ValueError(
            "--date-column must be supplied when --training-end is used."
        )
    if date_column not in dataframe.columns:
        raise KeyError(f"Date column not found: {date_column}")

    work = dataframe.copy()
    work[date_column] = pd.to_datetime(
        work[date_column],
        errors="coerce",
    )
    cutoff = pd.Timestamp(training_end)

    invalid_dates = work[date_column].isna()
    if invalid_dates.any():
        print(
            f"Warning: dropping {invalid_dates.sum()} rows with invalid dates.",
            file=sys.stderr,
        )

    work = work.loc[
        work[date_column].notna()
        & (work[date_column] <= cutoff)
    ].copy()

    if work.empty:
        raise ValueError(
            f"No rows remain on or before the training cutoff {cutoff.date()}."
        )

    return work


def choose_predictors(
    dataframe: pd.DataFrame,
    target: str,
    predictors: Optional[Sequence[str]],
    date_column: Optional[str],
    exclude_columns: Sequence[str],
) -> List[str]:
    """Use supplied predictors or automatically select eligible numeric columns."""
    if target not in dataframe.columns:
        raise KeyError(f"Target column not found: {target}")

    if predictors:
        selected = list(dict.fromkeys(predictors))
    else:
        excluded = {target, *exclude_columns}
        if date_column is not None:
            excluded.add(date_column)

        numeric_columns = dataframe.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        selected = [
            column
            for column in numeric_columns
            if column not in excluded
        ]

    missing = [
        column for column in selected
        if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(f"Predictor columns not found: {missing}")

    if target in selected:
        raise ValueError(
            "The target column must not also be included as a predictor."
        )

    if not selected:
        raise ValueError("No predictor columns were selected.")

    return selected


def prepare_modelling_data(
    dataframe: pd.DataFrame,
    predictors: Sequence[str],
    target: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Coerce predictors/target to numeric values and remove incomplete rows."""
    work = dataframe[list(predictors) + [target]].copy()

    for column in list(predictors) + [target]:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if len(work) < 10:
        raise ValueError(
            f"Only {len(work)} complete observations remain. "
            "At least 10 observations are recommended."
        )

    x = work[list(predictors)].to_numpy(dtype=float)
    y = work[target].to_numpy(dtype=float)

    if np.isclose(np.nanstd(y, ddof=1), 0):
        raise ValueError("The target column has no usable variation.")

    return work, x, y


# =============================================================================
# RANDOM FOREST TUNING
# =============================================================================

def build_default_random_forest(
    random_state: int,
    n_jobs: int,
) -> RandomForestRegressor:
    """Create the default Random Forest estimator used by Boruta."""
    return RandomForestRegressor(
        n_estimators=500,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def tune_random_forest(
    x: np.ndarray,
    y: np.ndarray,
    requested_cv: int,
    n_iter_search: int,
    random_state: int,
    n_jobs: int,
) -> Tuple[RandomForestRegressor, Dict[str, object]]:
    """Tune Random Forest hyperparameters using RandomizedSearchCV."""
    n_samples = len(y)

    cv_folds = min(requested_cv, n_samples)
    if cv_folds < 2:
        raise ValueError(
            "At least two observations are required for cross-validation."
        )

    estimator = RandomForestRegressor(
        random_state=random_state,
        n_jobs=n_jobs,
    )

    parameter_distributions = {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.75, 1.0],
    }

    maximum_combinations = math.prod(
        len(values)
        for values in parameter_distributions.values()
    )
    actual_n_iter = min(n_iter_search, maximum_combinations)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_distributions,
        n_iter=actual_n_iter,
        scoring="neg_mean_squared_error",
        cv=cv_folds,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
    search.fit(x, y)

    return search.best_estimator_, search.best_params_


# =============================================================================
# BORUTA FEATURE SELECTION
# =============================================================================

def run_boruta(
    estimator: RandomForestRegressor,
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    percentile: int,
    alpha: float,
    random_state: int,
) -> BorutaPy:
    """Fit Boruta using the configured Random Forest estimator."""
    selector = BorutaPy(
        estimator=estimator,
        n_estimators="auto",
        perc=percentile,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
    )
    selector.fit(x, y)
    return selector


def build_feature_table(
    predictors: Sequence[str],
    selector: BorutaPy,
    fitted_estimator: RandomForestRegressor,
    x: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    """Create a complete feature-selection and ranking table."""
    # Refit the tuned/default Random Forest on all available screening data to
    # provide a supplementary impurity-based importance value.
    fitted_estimator.fit(x, y)
    rf_importance = fitted_estimator.feature_importances_

    status = np.where(
        selector.support_,
        "Confirmed",
        np.where(
            selector.support_weak_,
            "Tentative",
            "Rejected",
        ),
    )

    table = pd.DataFrame(
        {
            "Feature": list(predictors),
            "Status": status,
            "Selected": selector.support_.astype(bool),
            "Tentative": selector.support_weak_.astype(bool),
            "Boruta rank raw": selector.ranking_.astype(int),
            "Random Forest importance": rf_importance,
        }
    )

    # Competition ranking: 1, 1, 3, ...
    table["Boruta rank"] = (
        table["Boruta rank raw"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    status_order = pd.CategoricalDtype(
        categories=["Confirmed", "Tentative", "Rejected"],
        ordered=True,
    )
    table["Status"] = table["Status"].astype(status_order)

    return (
        table.sort_values(
            ["Status", "Boruta rank", "Random Forest importance"],
            ascending=[True, True, False],
        )
        .reset_index(drop=True)
    )


# =============================================================================
# OUTPUTS
# =============================================================================

def safe_filename(text: str) -> str:
    """Convert a target name to a portable filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return cleaned.strip("_") or "target"


def save_feature_lists(
    feature_table: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Save confirmed, tentative, and rejected features as text files."""
    for status in ["Confirmed", "Tentative", "Rejected"]:
        features = feature_table.loc[
            feature_table["Status"] == status,
            "Feature",
        ].tolist()

        output_path = (
            output_directory
            / f"{status.lower()}_features.txt"
        )
        output_path.write_text(
            "\n".join(features) + ("\n" if features else ""),
            encoding="utf-8",
        )


def plot_feature_ranks(
    feature_table: pd.DataFrame,
    target: str,
    output_directory: Path,
) -> Path:
    """Create a horizontal feature-ranking plot using Matplotlib."""
    plot_table = feature_table.sort_values(
        ["Boruta rank", "Feature"],
        ascending=[False, True],
    )

    figure_height = max(4.5, 0.42 * len(plot_table) + 1.5)
    figure, axis = plt.subplots(
        figsize=(10, figure_height),
    )

    axis.barh(
        plot_table["Feature"],
        plot_table["Boruta rank"],
    )
    axis.set_xlabel("Boruta rank (lower is better)")
    axis.set_ylabel("Predictor")
    axis.set_title(f"Boruta feature ranks for {target}")
    axis.grid(axis="x", alpha=0.25)

    # Add status labels without imposing custom colours.
    for row_position, (_, row) in enumerate(plot_table.iterrows()):
        axis.text(
            row["Boruta rank"] + 0.1,
            row_position,
            str(row["Status"]),
            va="center",
            fontsize=9,
        )

    figure.tight_layout()

    output_path = (
        output_directory
        / f"boruta_feature_ranks_{safe_filename(target)}.png"
    )
    figure.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def save_outputs(
    feature_table: pd.DataFrame,
    best_parameters: Dict[str, object],
    target: str,
    predictors: Sequence[str],
    n_observations: int,
    output_directory: Path,
    create_plot: bool = True,
) -> None:
    """Save ranking tables, feature lists, metadata, and figure."""
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "boruta_feature_ranking.csv"
    excel_path = output_directory / "boruta_feature_ranking.xlsx"
    metadata_path = output_directory / "boruta_run_metadata.json"

    feature_table.to_csv(csv_path, index=False)
    feature_table.to_excel(excel_path, index=False)

    save_feature_lists(
        feature_table=feature_table,
        output_directory=output_directory,
    )

    metadata = {
        "target": target,
        "n_complete_observations": n_observations,
        "predictors_considered": list(predictors),
        "confirmed_features": feature_table.loc[
            feature_table["Status"] == "Confirmed",
            "Feature",
        ].tolist(),
        "tentative_features": feature_table.loc[
            feature_table["Status"] == "Tentative",
            "Feature",
        ].tolist(),
        "rejected_features": feature_table.loc[
            feature_table["Status"] == "Rejected",
            "Feature",
        ].tolist(),
        "random_forest_parameters": best_parameters,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Saved: {csv_path}")
    print(f"Saved: {excel_path}")
    print(f"Saved: {metadata_path}")

    if create_plot:
        plot_path = plot_feature_ranks(
            feature_table=feature_table,
            target=target,
            output_directory=output_directory,
        )
        print(f"Saved: {plot_path}")


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main() -> None:
    arguments = parse_arguments()

    sheet_name = normalize_sheet_name(arguments.sheet_name)

    dataframe = load_table(
        input_file=arguments.input_file,
        sheet_name=sheet_name,
    )
    dataframe = restrict_to_training_period(
        dataframe=dataframe,
        date_column=arguments.date_column,
        training_end=arguments.training_end,
    )

    predictors = choose_predictors(
        dataframe=dataframe,
        target=arguments.target,
        predictors=arguments.predictors,
        date_column=arguments.date_column,
        exclude_columns=arguments.exclude_columns,
    )

    clean_data, x, y = prepare_modelling_data(
        dataframe=dataframe,
        predictors=predictors,
        target=arguments.target,
    )

    print(f"Target: {arguments.target}")
    print(f"Predictors considered ({len(predictors)}): {predictors}")
    print(f"Complete observations used: {len(clean_data)}")

    if arguments.no_tuning:
        estimator = build_default_random_forest(
            random_state=arguments.random_state,
            n_jobs=arguments.n_jobs,
        )
        best_parameters: Dict[str, object] = estimator.get_params()
    else:
        estimator, best_parameters = tune_random_forest(
            x=x,
            y=y,
            requested_cv=arguments.cv,
            n_iter_search=arguments.n_iter_search,
            random_state=arguments.random_state,
            n_jobs=arguments.n_jobs,
        )
        print("Best Random Forest parameters:")
        print(best_parameters)

    selector = run_boruta(
        estimator=estimator,
        x=x,
        y=y,
        max_iter=arguments.boruta_max_iter,
        percentile=arguments.boruta_percentile,
        alpha=arguments.alpha,
        random_state=arguments.random_state,
    )

    feature_table = build_feature_table(
        predictors=predictors,
        selector=selector,
        fitted_estimator=estimator,
        x=x,
        y=y,
    )

    print("\nConfirmed features:")
    print(
        feature_table.loc[
            feature_table["Status"] == "Confirmed",
            "Feature",
        ].tolist()
    )
    print("\nTentative features:")
    print(
        feature_table.loc[
            feature_table["Status"] == "Tentative",
            "Feature",
        ].tolist()
    )

    save_outputs(
        feature_table=feature_table,
        best_parameters=best_parameters,
        target=arguments.target,
        predictors=predictors,
        n_observations=len(clean_data),
        output_directory=arguments.output_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Boruta feature selection failed: {error}", file=sys.stderr)
        raise
