"""
General-purpose dual-branch LSTM forward-chain forecasting workflow.

The workflow is designed for continuous water-quality targets such as:
- dissolved oxygen (DO);
- electrical conductivity (EC);
- turbidity;
- total suspended solids (TSS);
- any other numeric water-quality variable.

It trains the LSTM using reference/reanalysis predictors and produces:

1. Ex-post/reference-driven forecasts using target-month reference predictors.
2. Ex-ante/seasonal-forecast-driven forecasts using lead-specific predictors.

Only two evaluation metrics are reported:
- Kling-Gupta Efficiency (KGE)
- Normalized Mean Absolute Error (MAE%)

The code intentionally uses deterministic prediction. Monte Carlo dropout,
prediction intervals, R-squared, raw MAE, and RMSE are not calculated.

Run
---
python lstm_forward_chain.py --config lstm_forward_chain_config.json
"""

from __future__ import annotations

# Set deterministic TensorFlow-related environment variables before importing
# TensorFlow.
import os

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class DataConfig:
    reference_file: Path
    reference_sheet: object
    reference_date_columns: Tuple[str, ...]
    lead_directory: Path
    lead_file_pattern: str
    lead_sheet: object
    lead_date_columns: Tuple[str, ...]
    target: str
    history_features: Tuple[str, ...]
    future_features: Tuple[str, ...]


@dataclass(frozen=True)
class ForecastConfig:
    leads: Tuple[int, ...]
    lag_steps: Tuple[int, ...]
    lookback_months: int
    initial_train_start: str
    issue_start: str
    issue_end: str
    retrain_frequency: str
    fixed_train_end: Optional[str]
    nowcast_policy: str
    auto_trim_to_observations: bool


@dataclass(frozen=True)
class ModelConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    first_lstm_units: int
    second_lstm_units: int
    dropout_rate: float
    patience: int
    validation_fraction: float
    huber_delta: float
    random_seed: int
    target_scaling: bool
    clip_predictions_at_zero: bool
    verbose: int


@dataclass(frozen=True)
class OutputConfig:
    output_directory: Path
    run_name: str
    save_per_lead_workbook: bool
    save_models: bool


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    forecast: ForecastConfig
    model: ModelConfig
    output: OutputConfig


@dataclass
class DataBundle:
    target: pd.Series
    reference: pd.DataFrame
    forecast_by_lead: Dict[int, pd.DataFrame]


@dataclass
class FittedScalers:
    sequence: StandardScaler
    static: RobustScaler
    target: Optional[StandardScaler]


@dataclass
class FitBundle:
    model: keras.Model
    scalers: FittedScalers
    training_history: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    static_dimension: int


# =============================================================================
# CONFIGURATION LOADING AND VALIDATION
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic dual-branch LSTM forward-chain forecasting "
            "for a continuous water-quality target."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


def normalize_sheet_name(value: object) -> object:
    if isinstance(value, str) and re.fullmatch(r"\d+", value):
        return int(value)
    return value


def load_configuration(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    data_raw = raw["data"]
    forecast_raw = raw["forecast"]
    model_raw = raw["model"]
    output_raw = raw["output"]

    config = AppConfig(
        data=DataConfig(
            reference_file=Path(data_raw["reference_file"]),
            reference_sheet=normalize_sheet_name(
                data_raw.get("reference_sheet", 0)
            ),
            reference_date_columns=tuple(
                data_raw.get(
                    "reference_date_columns",
                    ["Month_Start", "Month"],
                )
            ),
            lead_directory=Path(data_raw["lead_directory"]),
            lead_file_pattern=data_raw[
                "lead_file_pattern"
            ],
            lead_sheet=normalize_sheet_name(
                data_raw.get("lead_sheet", 0)
            ),
            lead_date_columns=tuple(
                data_raw.get(
                    "lead_date_columns",
                    ["target_month", "Month_Start", "Month"],
                )
            ),
            target=str(data_raw["target"]),
            history_features=tuple(
                data_raw["history_features"]
            ),
            future_features=tuple(
                data_raw["future_features"]
            ),
        ),
        forecast=ForecastConfig(
            leads=tuple(
                int(value) for value in forecast_raw["leads"]
            ),
            lag_steps=tuple(
                int(value) for value in forecast_raw["lag_steps"]
            ),
            lookback_months=int(
                forecast_raw["lookback_months"]
            ),
            initial_train_start=str(
                forecast_raw["initial_train_start"]
            ),
            issue_start=str(forecast_raw["issue_start"]),
            issue_end=str(forecast_raw["issue_end"]),
            retrain_frequency=str(
                forecast_raw.get(
                    "retrain_frequency",
                    "monthly",
                )
            ).lower(),
            fixed_train_end=forecast_raw.get(
                "fixed_train_end"
            ),
            nowcast_policy=str(
                forecast_raw.get(
                    "nowcast_policy",
                    "persistence",
                )
            ).lower(),
            auto_trim_to_observations=bool(
                forecast_raw.get(
                    "auto_trim_to_observations",
                    True,
                )
            ),
        ),
        model=ModelConfig(
            epochs=int(model_raw.get("epochs", 250)),
            batch_size=int(model_raw.get("batch_size", 8)),
            learning_rate=float(
                model_raw.get("learning_rate", 1e-3)
            ),
            first_lstm_units=int(
                model_raw.get("first_lstm_units", 32)
            ),
            second_lstm_units=int(
                model_raw.get("second_lstm_units", 16)
            ),
            dropout_rate=float(
                model_raw.get("dropout_rate", 0.30)
            ),
            patience=int(model_raw.get("patience", 15)),
            validation_fraction=float(
                model_raw.get("validation_fraction", 0.20)
            ),
            huber_delta=float(
                model_raw.get("huber_delta", 1.0)
            ),
            random_seed=int(
                model_raw.get("random_seed", 42)
            ),
            target_scaling=bool(
                model_raw.get("target_scaling", False)
            ),
            clip_predictions_at_zero=bool(
                model_raw.get(
                    "clip_predictions_at_zero",
                    False,
                )
            ),
            verbose=int(model_raw.get("verbose", 0)),
        ),
        output=OutputConfig(
            output_directory=Path(
                output_raw.get(
                    "output_directory",
                    "outputs/lstm_forward_chain",
                )
            ),
            run_name=str(
                output_raw.get(
                    "run_name",
                    "lstm_forward_chain",
                )
            ),
            save_per_lead_workbook=bool(
                output_raw.get(
                    "save_per_lead_workbook",
                    True,
                )
            ),
            save_models=bool(
                output_raw.get("save_models", False)
            ),
        ),
    )

    validate_configuration(config)
    return config


def validate_configuration(config: AppConfig) -> None:
    leads = sorted(set(config.forecast.leads))
    lag_steps = sorted(set(config.forecast.lag_steps))

    if not leads:
        raise ValueError("At least one forecast lead is required.")

    expected_leads = list(range(1, max(leads) + 1))
    if leads != expected_leads:
        raise ValueError(
            "Recursive forecasting requires consecutive leads starting at 1. "
            f"Received {leads}; expected {expected_leads}."
        )

    if not lag_steps or any(lag <= 0 for lag in lag_steps):
        raise ValueError(
            "lag_steps must contain one or more positive integers."
        )

    if config.forecast.lookback_months < 1:
        raise ValueError("lookback_months must be at least 1.")

    if "{lead}" not in config.data.lead_file_pattern:
        raise ValueError(
            "lead_file_pattern must contain the placeholder '{lead}'."
        )

    if not config.data.history_features:
        raise ValueError("history_features cannot be empty.")

    if not config.data.future_features:
        raise ValueError("future_features cannot be empty.")

    allowed_retraining = {"monthly", "yearly", "fixed"}
    if config.forecast.retrain_frequency not in allowed_retraining:
        raise ValueError(
            "retrain_frequency must be monthly, yearly, or fixed."
        )

    if (
        config.forecast.retrain_frequency == "fixed"
        and not config.forecast.fixed_train_end
    ):
        raise ValueError(
            "fixed_train_end is required when retrain_frequency is fixed."
        )

    if config.forecast.nowcast_policy not in {
        "persistence",
        "observed",
    }:
        raise ValueError(
            "nowcast_policy must be persistence or observed."
        )

    if not 0 <= config.model.validation_fraction < 0.5:
        raise ValueError(
            "validation_fraction must be in the interval [0, 0.5)."
        )

    if not 0 <= config.model.dropout_rate < 1:
        raise ValueError("dropout_rate must be in the interval [0, 1).")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Older TensorFlow builds may not provide this function.
        pass

    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        # Threading configuration cannot be changed after runtime
        # initialization.
        pass


# =============================================================================
# DATE AND FILE HELPERS
# =============================================================================

def to_month(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Period(value, freq="M").to_timestamp()


def add_months(
    value: str | pd.Timestamp,
    number_of_months: int,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp
        + pd.DateOffset(months=number_of_months)
    ).to_period("M").to_timestamp()


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return cleaned.strip("_") or "target"


def read_monthly_table(
    path: Path,
    sheet_name: object,
    date_column_candidates: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(
            path,
            sheet_name=sheet_name,
        )
    elif suffix == ".csv":
        dataframe = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported input format for {path}. "
            "Use .xlsx, .xls, or .csv."
        )

    date_column = next(
        (
            column
            for column in date_column_candidates
            if column in dataframe.columns
        ),
        None,
    )
    if date_column is None:
        raise KeyError(
            f"No monthly date column was found in {path}. "
            f"Expected one of {list(date_column_candidates)}."
        )

    dataframe[date_column] = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
    ).dt.to_period("M").dt.to_timestamp()

    invalid_date_count = int(dataframe[date_column].isna().sum())
    if invalid_date_count:
        raise ValueError(
            f"{path} contains {invalid_date_count} invalid monthly dates."
        )

    dataframe = (
        dataframe.sort_values(date_column)
        .set_index(date_column)
    )
    dataframe.index.name = "Month_Start"

    duplicated = dataframe.index.duplicated(keep=False)
    if duplicated.any():
        duplicate_months = (
            dataframe.index[duplicated]
            .strftime("%Y-%m")
            .unique()
            .tolist()
        )
        raise ValueError(
            f"{path} contains duplicate months: {duplicate_months[:10]}"
        )

    return dataframe


def numeric_subset(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    context: str,
) -> pd.DataFrame:
    missing = [
        column for column in columns
        if column not in dataframe.columns
    ]
    if missing:
        raise KeyError(f"{context}: missing columns {missing}")

    output = dataframe[list(columns)].copy()
    for column in columns:
        output[column] = pd.to_numeric(
            output[column],
            errors="coerce",
        )

    return output


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: AppConfig) -> DataBundle:
    reference_all = read_monthly_table(
        path=config.data.reference_file,
        sheet_name=config.data.reference_sheet,
        date_column_candidates=config.data.reference_date_columns,
    )

    required_reference_columns = list(
        dict.fromkeys(
            [
                config.data.target,
                *config.data.history_features,
                *config.data.future_features,
            ]
        )
    )

    reference_numeric = numeric_subset(
        dataframe=reference_all,
        columns=required_reference_columns,
        context="Reference/reanalysis data",
    )

    target = reference_numeric[
        config.data.target
    ].rename(config.data.target)

    forecast_by_lead: Dict[int, pd.DataFrame] = {}

    for lead in config.forecast.leads:
        path = (
            config.data.lead_directory
            / config.data.lead_file_pattern.format(
                lead=lead
            )
        )

        lead_table = read_monthly_table(
            path=path,
            sheet_name=config.data.lead_sheet,
            date_column_candidates=config.data.lead_date_columns,
        )

        forecast_by_lead[lead] = numeric_subset(
            dataframe=lead_table,
            columns=config.data.future_features,
            context=f"Lead {lead} forecast data",
        )

    print(
        "Reference-data span: "
        f"{reference_numeric.index.min().date()} to "
        f"{reference_numeric.index.max().date()} "
        f"(n={len(reference_numeric)})"
    )
    print(
        "Last available observed target month: "
        f"{target.dropna().index.max().date()}"
    )

    return DataBundle(
        target=target,
        reference=reference_numeric,
        forecast_by_lead=forecast_by_lead,
    )


# =============================================================================
# METRICS
# =============================================================================

def paired_finite_values(
    observed: Sequence[float],
    predicted: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)

    valid = (
        np.isfinite(observed_array)
        & np.isfinite(predicted_array)
    )
    return observed_array[valid], predicted_array[valid]


def kling_gupta_efficiency(
    observed: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """
    Calculate KGE using correlation, variability ratio, and bias ratio.

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    """
    y_true, y_pred = paired_finite_values(
        observed,
        predicted,
    )

    if len(y_true) < 2:
        return np.nan

    observed_std = np.std(y_true, ddof=1)
    predicted_std = np.std(y_pred, ddof=1)
    observed_mean = np.mean(y_true)

    if (
        np.isclose(observed_std, 0)
        or np.isclose(predicted_std, 0)
        or np.isclose(observed_mean, 0)
    ):
        return np.nan

    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    if not np.isfinite(correlation):
        return np.nan

    alpha = predicted_std / observed_std
    beta = np.mean(y_pred) / observed_mean

    return float(
        1.0
        - np.sqrt(
            (correlation - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def normalized_mae_percent(
    observed: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """
    Calculate MAE% as:

        100 * mean(abs(predicted - observed)) / mean(abs(observed))

    For non-negative water-quality variables, this is equivalent to
    normalizing by the mean observed value.
    """
    y_true, y_pred = paired_finite_values(
        observed,
        predicted,
    )

    if len(y_true) == 0:
        return np.nan

    denominator = np.mean(np.abs(y_true))
    if np.isclose(denominator, 0):
        return np.nan

    return float(
        100.0
        * np.mean(np.abs(y_pred - y_true))
        / denominator
    )


# =============================================================================
# FEATURE ASSEMBLY
# =============================================================================

def cyclical_month_features(
    target_month: pd.Timestamp,
) -> np.ndarray:
    month_number = target_month.month
    return np.asarray(
        [
            np.sin(
                2.0 * np.pi * month_number / 12.0
            ),
            np.cos(
                2.0 * np.pi * month_number / 12.0
            ),
        ],
        dtype=float,
    )


def lead_one_hot(
    lead: int,
    all_leads: Sequence[int],
) -> np.ndarray:
    vector = np.zeros(len(all_leads), dtype=float)
    vector[list(all_leads).index(lead)] = 1.0
    return vector


def historical_sequence(
    reference: pd.DataFrame,
    history_features: Sequence[str],
    issue_month: pd.Timestamp,
    lookback_months: int,
) -> pd.DataFrame:
    months = [
        add_months(
            issue_month,
            offset,
        )
        for offset in range(
            -(lookback_months - 1),
            1,
        )
    ]

    missing = [
        month for month in months
        if month not in reference.index
    ]
    if missing:
        raise KeyError(
            "Missing historical predictor months ending "
            f"{issue_month.date()}: "
            f"{[month.strftime('%Y-%m') for month in missing]}"
        )

    sequence = reference.loc[
        months,
        list(history_features),
    ]

    if sequence.isna().any().any():
        raise ValueError(
            "Historical predictors contain missing values for "
            f"issue month {issue_month.strftime('%Y-%m')}."
        )

    return sequence


def reference_future_predictors(
    bundle: DataBundle,
    target_month: pd.Timestamp,
    future_features: Sequence[str],
) -> np.ndarray:
    if target_month not in bundle.reference.index:
        raise KeyError(
            "Reference predictors are unavailable for target month "
            f"{target_month.strftime('%Y-%m')}."
        )

    values = bundle.reference.loc[
        target_month,
        list(future_features),
    ]

    if values.isna().any():
        raise ValueError(
            "Reference future predictors contain missing values for "
            f"{target_month.strftime('%Y-%m')}."
        )

    return values.to_numpy(dtype=float)


def seasonal_future_predictors(
    bundle: DataBundle,
    lead: int,
    target_month: pd.Timestamp,
    future_features: Sequence[str],
) -> np.ndarray:
    table = bundle.forecast_by_lead[lead]

    if target_month not in table.index:
        raise KeyError(
            f"Lead {lead} forecast predictors are unavailable for "
            f"{target_month.strftime('%Y-%m')}."
        )

    values = table.loc[
        target_month,
        list(future_features),
    ]

    if values.isna().any():
        raise ValueError(
            f"Lead {lead} future predictors contain missing values for "
            f"{target_month.strftime('%Y-%m')}."
        )

    return values.to_numpy(dtype=float)


def compute_nowcast(
    target: pd.Series,
    issue_month: pd.Timestamp,
    policy: str,
) -> float:
    if policy == "persistence":
        source_month = add_months(issue_month, -1)
    elif policy == "observed":
        source_month = issue_month
    else:
        raise ValueError(
            "nowcast policy must be persistence or observed."
        )

    if source_month not in target.index:
        raise KeyError(
            "Nowcast source month is unavailable: "
            f"{source_month.strftime('%Y-%m')}"
        )

    value = target.loc[source_month]
    if not np.isfinite(value):
        raise ValueError(
            "Nowcast source target is missing for "
            f"{source_month.strftime('%Y-%m')}."
        )

    return float(value)


def training_lag_vector(
    target: pd.Series,
    issue_month: pd.Timestamp,
    target_month: pd.Timestamp,
    lag_steps: Sequence[int],
    nowcast_policy: str,
) -> np.ndarray:
    """
    Construct lagged target inputs for model training.

    Values before the issue month are observed. The issue-month value follows
    the configured nowcast policy. Values after the issue month use observed
    training-period targets as teacher forcing. During forecasting, those
    post-issue values are replaced recursively by prior model predictions.
    """
    nowcast = compute_nowcast(
        target,
        issue_month,
        nowcast_policy,
    )

    values: List[float] = []

    for lag in lag_steps:
        lag_month = add_months(target_month, -lag)

        if lag_month == issue_month:
            value = nowcast
        else:
            if lag_month not in target.index:
                raise KeyError(
                    "Training lag target is unavailable for "
                    f"{lag_month.strftime('%Y-%m')}."
                )
            value = target.loc[lag_month]

        if not np.isfinite(value):
            raise ValueError(
                "Training lag target is missing for "
                f"{lag_month.strftime('%Y-%m')}."
            )

        values.append(float(value))

    return np.asarray(values, dtype=float)


def recursive_forecast_lag_vector(
    target: pd.Series,
    issue_month: pd.Timestamp,
    target_month: pd.Timestamp,
    lag_steps: Sequence[int],
    prediction_cache: Mapping[pd.Timestamp, float],
    nowcast_policy: str,
) -> np.ndarray:
    """
    Construct lagged target inputs without using future observations.

    - lag month before issue month: observed target;
    - lag month equal to issue month: configured nowcast;
    - lag month after issue month: earlier recursive prediction.
    """
    nowcast = compute_nowcast(
        target,
        issue_month,
        nowcast_policy,
    )

    values: List[float] = []

    for lag in lag_steps:
        lag_month = add_months(target_month, -lag)

        if lag_month < issue_month:
            if lag_month not in target.index:
                raise KeyError(
                    "Observed lag target is unavailable for "
                    f"{lag_month.strftime('%Y-%m')}."
                )
            value = target.loc[lag_month]

        elif lag_month == issue_month:
            value = nowcast

        else:
            if lag_month not in prediction_cache:
                raise KeyError(
                    "Recursive prediction is unavailable for lag month "
                    f"{lag_month.strftime('%Y-%m')}. "
                    "Check that leads are consecutive and processed in order."
                )
            value = prediction_cache[lag_month]

        if not np.isfinite(value):
            raise ValueError(
                "Lagged target value is non-finite for "
                f"{lag_month.strftime('%Y-%m')}."
            )

        values.append(float(value))

    return np.asarray(values, dtype=float)


# =============================================================================
# TRAINING SAMPLE GENERATION
# =============================================================================

def first_training_issue_month(
    train_start: pd.Timestamp,
    lookback_months: int,
    lag_steps: Sequence[int],
) -> pd.Timestamp:
    required_offset = max(
        lookback_months - 1,
        max(lag_steps) - 1,
        1,
    )
    return add_months(train_start, required_offset)


def collect_training_samples(
    bundle: DataBundle,
    config: AppConfig,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    first_issue = first_training_issue_month(
        train_start=train_start,
        lookback_months=config.forecast.lookback_months,
        lag_steps=config.forecast.lag_steps,
    )

    issue_months = pd.period_range(
        start=first_issue,
        end=train_end,
        freq="M",
    ).to_timestamp()

    sequence_samples: List[np.ndarray] = []
    static_samples: List[np.ndarray] = []
    targets: List[float] = []
    metadata_rows: List[Dict[str, object]] = []

    for issue_month in issue_months:
        history = historical_sequence(
            reference=bundle.reference,
            history_features=config.data.history_features,
            issue_month=issue_month,
            lookback_months=config.forecast.lookback_months,
        )

        for lead in config.forecast.leads:
            target_month = add_months(
                issue_month,
                lead,
            )

            if target_month > train_end:
                continue

            if target_month not in bundle.target.index:
                continue

            observed_target = bundle.target.loc[target_month]
            if not np.isfinite(observed_target):
                continue

            future_values = reference_future_predictors(
                bundle=bundle,
                target_month=target_month,
                future_features=config.data.future_features,
            )

            lag_values = training_lag_vector(
                target=bundle.target,
                issue_month=issue_month,
                target_month=target_month,
                lag_steps=config.forecast.lag_steps,
                nowcast_policy=config.forecast.nowcast_policy,
            )

            static_values = np.concatenate(
                [
                    future_values,
                    lag_values,
                    lead_one_hot(
                        lead,
                        config.forecast.leads,
                    ),
                    cyclical_month_features(
                        target_month
                    ),
                ]
            )

            sequence_samples.append(
                history.to_numpy(dtype=float)
            )
            static_samples.append(static_values)
            targets.append(float(observed_target))
            metadata_rows.append(
                {
                    "IssueMonth": issue_month,
                    "TargetMonth": target_month,
                    "Lead": lead,
                }
            )

    if not targets:
        raise ValueError(
            "No valid training samples were generated for "
            f"{train_start.strftime('%Y-%m')} to "
            f"{train_end.strftime('%Y-%m')}."
        )

    metadata = pd.DataFrame(metadata_rows)

    # Samples are already generated in chronological issue-month order, but
    # sorting explicitly ensures a leakage-aware chronological validation split.
    order = (
        metadata.sort_values(
            ["IssueMonth", "Lead"]
        ).index.to_numpy()
    )

    return (
        np.stack(sequence_samples)[order],
        np.stack(static_samples)[order],
        np.asarray(targets, dtype=float)[order],
        metadata.loc[order].reset_index(drop=True),
    )


# =============================================================================
# MODEL
# =============================================================================

def build_lstm_model(
    number_of_history_features: int,
    lookback_months: int,
    number_of_static_features: int,
    model_config: ModelConfig,
) -> keras.Model:
    sequence_input = keras.Input(
        shape=(
            lookback_months,
            number_of_history_features,
        ),
        name="historical_sequence",
    )

    sequence_branch = layers.Masking(
        name="sequence_masking"
    )(sequence_input)

    sequence_branch = layers.LSTM(
        model_config.first_lstm_units,
        return_sequences=True,
        name="lstm_1",
    )(sequence_branch)

    sequence_branch = layers.LSTM(
        model_config.second_lstm_units,
        return_sequences=False,
        name="lstm_2",
    )(sequence_branch)

    sequence_branch = layers.Dense(
        32,
        activation="relu",
        name="historical_dense",
    )(sequence_branch)

    sequence_branch = layers.Dropout(
        model_config.dropout_rate,
        name="historical_dropout",
    )(sequence_branch)

    static_input = keras.Input(
        shape=(number_of_static_features,),
        name="future_and_lag_features",
    )

    merged = layers.Concatenate(
        name="feature_concatenation"
    )(
        [
            sequence_branch,
            static_input,
        ]
    )

    merged = layers.Dense(
        64,
        activation="relu",
        name="merged_dense_1",
    )(merged)

    merged = layers.Dropout(
        model_config.dropout_rate,
        name="merged_dropout",
    )(merged)

    merged = layers.Dense(
        32,
        activation="relu",
        name="merged_dense_2",
    )(merged)

    output = layers.Dense(
        1,
        activation="linear",
        name="water_quality_prediction",
    )(merged)

    model = keras.Model(
        inputs=[
            sequence_input,
            static_input,
        ],
        outputs=output,
        name="dual_input_lstm",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=model_config.learning_rate
        ),
        loss=keras.losses.Huber(
            delta=model_config.huber_delta
        ),
    )

    return model


def scale_training_data(
    sequence_samples: np.ndarray,
    static_samples: np.ndarray,
    targets: np.ndarray,
    target_scaling: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    FittedScalers,
]:
    number_of_history_features = sequence_samples.shape[-1]

    sequence_scaler = StandardScaler()
    sequence_scaler.fit(
        sequence_samples.reshape(
            -1,
            number_of_history_features,
        )
    )

    sequence_scaled = (
        sequence_scaler.transform(
            sequence_samples.reshape(
                -1,
                number_of_history_features,
            )
        )
        .reshape(sequence_samples.shape)
    )

    static_scaler = RobustScaler()
    static_scaled = static_scaler.fit_transform(
        static_samples
    )

    target_scaler: Optional[StandardScaler]

    if target_scaling:
        target_scaler = StandardScaler()
        target_scaled = (
            target_scaler
            .fit_transform(
                targets.reshape(-1, 1)
            )
            .reshape(-1)
        )
    else:
        target_scaler = None
        target_scaled = targets.copy()

    return (
        sequence_scaled,
        static_scaled,
        target_scaled,
        FittedScalers(
            sequence=sequence_scaler,
            static=static_scaler,
            target=target_scaler,
        ),
    )


def chronological_train_validation_split(
    sequence_samples: np.ndarray,
    static_samples: np.ndarray,
    targets: np.ndarray,
    validation_fraction: float,
) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray,
    Optional[Tuple[Dict[str, np.ndarray], np.ndarray]],
]:
    number_of_samples = len(targets)

    if (
        validation_fraction <= 0
        or number_of_samples < 10
    ):
        training_inputs = {
            "historical_sequence": sequence_samples,
            "future_and_lag_features": static_samples,
        }
        return training_inputs, targets, None

    validation_count = max(
        1,
        int(round(
            number_of_samples
            * validation_fraction
        )),
    )
    split_index = number_of_samples - validation_count

    if split_index < 2:
        training_inputs = {
            "historical_sequence": sequence_samples,
            "future_and_lag_features": static_samples,
        }
        return training_inputs, targets, None

    training_inputs = {
        "historical_sequence": sequence_samples[
            :split_index
        ],
        "future_and_lag_features": static_samples[
            :split_index
        ],
    }
    training_targets = targets[:split_index]

    validation_inputs = {
        "historical_sequence": sequence_samples[
            split_index:
        ],
        "future_and_lag_features": static_samples[
            split_index:
        ],
    }
    validation_targets = targets[split_index:]

    return (
        training_inputs,
        training_targets,
        (
            validation_inputs,
            validation_targets,
        ),
    )


def train_model(
    bundle: DataBundle,
    config: AppConfig,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> FitBundle:
    (
        sequence_samples,
        static_samples,
        targets,
        metadata,
    ) = collect_training_samples(
        bundle=bundle,
        config=config,
        train_start=train_start,
        train_end=train_end,
    )

    (
        sequence_scaled,
        static_scaled,
        target_scaled,
        scalers,
    ) = scale_training_data(
        sequence_samples=sequence_samples,
        static_samples=static_samples,
        targets=targets,
        target_scaling=config.model.target_scaling,
    )

    (
        training_inputs,
        training_targets,
        validation_data,
    ) = chronological_train_validation_split(
        sequence_samples=sequence_scaled,
        static_samples=static_scaled,
        targets=target_scaled,
        validation_fraction=config.model.validation_fraction,
    )

    # Reinitialize the TensorFlow random state for each independently trained
    # forward-chain model.
    set_random_seeds(config.model.random_seed)

    model = build_lstm_model(
        number_of_history_features=len(
            config.data.history_features
        ),
        lookback_months=config.forecast.lookback_months,
        number_of_static_features=static_samples.shape[1],
        model_config=config.model,
    )

    monitor = (
        "val_loss"
        if validation_data is not None
        else "loss"
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=config.model.patience,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x=training_inputs,
        y=training_targets,
        validation_data=validation_data,
        epochs=config.model.epochs,
        batch_size=min(
            config.model.batch_size,
            len(training_targets),
        ),
        shuffle=False,
        verbose=config.model.verbose,
        callbacks=callbacks,
    )

    history_table = pd.DataFrame(history.history)
    history_table.insert(
        0,
        "Epoch",
        np.arange(
            1,
            len(history_table) + 1,
        ),
    )
    history_table["TrainStart"] = train_start
    history_table["TrainEnd"] = train_end
    history_table["TrainingSamples"] = len(
        training_targets
    )
    history_table["TotalSamples"] = len(targets)
    history_table["LastTrainingIssue"] = (
        metadata["IssueMonth"].max()
    )

    return FitBundle(
        model=model,
        scalers=scalers,
        training_history=history_table,
        train_start=train_start,
        train_end=train_end,
        static_dimension=static_samples.shape[1],
    )


# =============================================================================
# DETERMINISTIC FORECASTING
# =============================================================================

def transform_sequence(
    sequence: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    values = sequence.to_numpy(dtype=float)
    return scaler.transform(values)


def inverse_target_prediction(
    prediction: np.ndarray,
    target_scaler: Optional[StandardScaler],
) -> np.ndarray:
    prediction = np.asarray(
        prediction,
        dtype=float,
    ).reshape(-1, 1)

    if target_scaler is None:
        return prediction.reshape(-1)

    return (
        target_scaler.inverse_transform(
            prediction
        )
        .reshape(-1)
    )


def deterministic_predict(
    fit: FitBundle,
    sequence_scaled: np.ndarray,
    static_scaled: np.ndarray,
    clip_at_zero: bool,
) -> float:
    prediction_scaled = fit.model.predict(
        {
            "historical_sequence": sequence_scaled,
            "future_and_lag_features": static_scaled,
        },
        verbose=0,
    ).reshape(-1)

    prediction = inverse_target_prediction(
        prediction_scaled,
        fit.scalers.target,
    )

    value = float(prediction[0])

    if clip_at_zero:
        value = max(0.0, value)

    return value


def forecast_one_issue(
    bundle: DataBundle,
    config: AppConfig,
    fit: FitBundle,
    issue_month: pd.Timestamp,
    allowed_leads: Sequence[int],
    branch: str,
) -> pd.DataFrame:
    history = historical_sequence(
        reference=bundle.reference,
        history_features=config.data.history_features,
        issue_month=issue_month,
        lookback_months=config.forecast.lookback_months,
    )

    history_scaled = transform_sequence(
        sequence=history,
        scaler=fit.scalers.sequence,
    ).reshape(
        1,
        config.forecast.lookback_months,
        len(config.data.history_features),
    )

    prediction_cache: Dict[pd.Timestamp, float] = {}
    records: List[Dict[str, object]] = []

    for lead in allowed_leads:
        target_month = add_months(
            issue_month,
            lead,
        )

        lag_values = recursive_forecast_lag_vector(
            target=bundle.target,
            issue_month=issue_month,
            target_month=target_month,
            lag_steps=config.forecast.lag_steps,
            prediction_cache=prediction_cache,
            nowcast_policy=config.forecast.nowcast_policy,
        )

        if branch == "Reference":
            future_values = reference_future_predictors(
                bundle=bundle,
                target_month=target_month,
                future_features=config.data.future_features,
            )
        elif branch == "SeasonalForecast":
            future_values = seasonal_future_predictors(
                bundle=bundle,
                lead=lead,
                target_month=target_month,
                future_features=config.data.future_features,
            )
        else:
            raise ValueError(
                "branch must be Reference or SeasonalForecast."
            )

        static_values = np.concatenate(
            [
                future_values,
                lag_values,
                lead_one_hot(
                    lead,
                    config.forecast.leads,
                ),
                cyclical_month_features(
                    target_month
                ),
            ]
        ).reshape(1, -1)

        static_scaled = fit.scalers.static.transform(
            static_values
        )

        predicted_target = deterministic_predict(
            fit=fit,
            sequence_scaled=history_scaled,
            static_scaled=static_scaled,
            clip_at_zero=(
                config.model.clip_predictions_at_zero
            ),
        )

        prediction_cache[
            target_month
        ] = predicted_target

        observed_target = (
            float(bundle.target.loc[target_month])
            if (
                target_month in bundle.target.index
                and np.isfinite(
                    bundle.target.loc[target_month]
                )
            )
            else np.nan
        )

        records.append(
            {
                "IssueMonth": issue_month,
                "TargetMonth": target_month,
                "Lead": lead,
                "Branch": branch,
                "Observed": observed_target,
                "Predicted": predicted_target,
                "TrainingStart": fit.train_start,
                "TrainingEnd": fit.train_end,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# FORWARD-CHAIN SCHEDULE
# =============================================================================

def training_end_for_issue(
    issue_month: pd.Timestamp,
    forecast_config: ForecastConfig,
) -> pd.Timestamp:
    if forecast_config.retrain_frequency == "monthly":
        return add_months(issue_month, -1)

    if forecast_config.retrain_frequency == "yearly":
        return pd.Timestamp(
            year=issue_month.year - 1,
            month=12,
            day=1,
        )

    if forecast_config.retrain_frequency == "fixed":
        return to_month(
            str(forecast_config.fixed_train_end)
        )

    raise ValueError(
        "Unsupported retrain_frequency."
    )


def allowed_leads_for_issue(
    issue_month: pd.Timestamp,
    nominal_leads: Sequence[int],
    last_observed_month: pd.Timestamp,
    auto_trim: bool,
) -> List[int]:
    if not auto_trim:
        return list(nominal_leads)

    return [
        lead
        for lead in nominal_leads
        if add_months(
            issue_month,
            lead,
        ) <= last_observed_month
    ]


def write_model_summary(
    model: keras.Model,
    output_path: Path,
) -> None:
    lines: List[str] = []
    model.summary(print_fn=lines.append)
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_forward_chain(
    bundle: DataBundle,
    config: AppConfig,
) -> pd.DataFrame:
    output_directory = config.output.output_directory
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    histories_directory = (
        output_directory
        / "training_histories"
    )
    histories_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    models_directory = (
        output_directory
        / "trained_models"
    )
    if config.output.save_models:
        models_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    issue_months = pd.period_range(
        start=config.forecast.issue_start,
        end=config.forecast.issue_end,
        freq="M",
    ).to_timestamp()

    train_start = to_month(
        config.forecast.initial_train_start
    )
    last_observed_month = (
        bundle.target.dropna().index.max()
    )

    all_forecasts: List[pd.DataFrame] = []
    reusable_fit_cache: Dict[
        pd.Timestamp,
        FitBundle,
    ] = {}

    architecture_saved = False

    for issue_month in issue_months:
        allowed_leads = allowed_leads_for_issue(
            issue_month=issue_month,
            nominal_leads=config.forecast.leads,
            last_observed_month=last_observed_month,
            auto_trim=(
                config.forecast
                .auto_trim_to_observations
            ),
        )

        if not allowed_leads:
            print(
                "Skipping issue month "
                f"{issue_month.strftime('%Y-%m')}: "
                "no target months remain within the observed horizon."
            )
            continue

        train_end = training_end_for_issue(
            issue_month=issue_month,
            forecast_config=config.forecast,
        )

        if train_end < train_start:
            raise ValueError(
                "Training end precedes training start for issue month "
                f"{issue_month.strftime('%Y-%m')}."
            )

        use_cache = (
            config.forecast.retrain_frequency
            in {"yearly", "fixed"}
        )

        if use_cache and train_end in reusable_fit_cache:
            fit = reusable_fit_cache[train_end]
        else:
            print(
                "Training model for "
                f"{train_start.strftime('%Y-%m')} to "
                f"{train_end.strftime('%Y-%m')}..."
            )

            fit = train_model(
                bundle=bundle,
                config=config,
                train_start=train_start,
                train_end=train_end,
            )

            history_path = (
                histories_directory
                / (
                    "training_history_"
                    f"{train_end.strftime('%Y_%m')}.csv"
                )
            )
            fit.training_history.to_csv(
                history_path,
                index=False,
            )

            if not architecture_saved:
                write_model_summary(
                    fit.model,
                    output_directory
                    / "model_architecture.txt",
                )
                architecture_saved = True

            if config.output.save_models:
                model_path = (
                    models_directory
                    / (
                        "lstm_model_"
                        f"{train_end.strftime('%Y_%m')}.keras"
                    )
                )
                fit.model.save(model_path)

            if use_cache:
                reusable_fit_cache[train_end] = fit

        reference_forecasts = forecast_one_issue(
            bundle=bundle,
            config=config,
            fit=fit,
            issue_month=issue_month,
            allowed_leads=allowed_leads,
            branch="Reference",
        )

        seasonal_forecasts = forecast_one_issue(
            bundle=bundle,
            config=config,
            fit=fit,
            issue_month=issue_month,
            allowed_leads=allowed_leads,
            branch="SeasonalForecast",
        )

        all_forecasts.extend(
            [
                reference_forecasts,
                seasonal_forecasts,
            ]
        )

        print(
            f"Issue {issue_month.strftime('%Y-%m')} | "
            f"training end {train_end.strftime('%Y-%m')} | "
            f"leads {allowed_leads}"
        )

    if not all_forecasts:
        raise ValueError(
            "No forecasts were generated."
        )

    return pd.concat(
        all_forecasts,
        ignore_index=True,
    )


# =============================================================================
# METRIC SUMMARIES AND OUTPUTS
# =============================================================================

def summarize_metrics(
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for branch in sorted(
        forecasts["Branch"].dropna().unique()
    ):
        branch_data = forecasts[
            forecasts["Branch"] == branch
        ]

        for lead in sorted(
            branch_data["Lead"].dropna().unique()
        ):
            subset = branch_data[
                branch_data["Lead"] == lead
            ]

            observed, predicted = paired_finite_values(
                subset["Observed"],
                subset["Predicted"],
            )

            rows.append(
                {
                    "Branch": branch,
                    "Lead": int(lead),
                    "N": int(len(observed)),
                    "KGE": kling_gupta_efficiency(
                        observed,
                        predicted,
                    ),
                    "MAE (%)": normalized_mae_percent(
                        observed,
                        predicted,
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["Branch", "Lead"])
        .reset_index(drop=True)
    )


def build_per_lead_tables(
    forecasts: pd.DataFrame,
) -> Dict[int, pd.DataFrame]:
    output: Dict[int, pd.DataFrame] = {}

    for lead in sorted(
        forecasts["Lead"].unique()
    ):
        subset = forecasts[
            forecasts["Lead"] == lead
        ]

        predictions = subset.pivot_table(
            index=[
                "IssueMonth",
                "TargetMonth",
            ],
            columns="Branch",
            values="Predicted",
            aggfunc="first",
        ).rename(
            columns={
                "Reference": "Predicted (Reference)",
                "SeasonalForecast":
                    "Predicted (Seasonal Forecast)",
            }
        )

        observed = (
            subset.groupby(
                [
                    "IssueMonth",
                    "TargetMonth",
                ]
            )["Observed"]
            .first()
            .rename("Observed")
        )

        table = pd.concat(
            [
                observed,
                predictions,
            ],
            axis=1,
        ).reset_index()

        expected_columns = [
            "IssueMonth",
            "TargetMonth",
            "Observed",
            "Predicted (Reference)",
            "Predicted (Seasonal Forecast)",
        ]

        for column in expected_columns:
            if column not in table.columns:
                table[column] = np.nan

        output[int(lead)] = (
            table[expected_columns]
            .sort_values(
                [
                    "TargetMonth",
                    "IssueMonth",
                ]
            )
            .reset_index(drop=True)
        )

    return output


def save_per_lead_workbook(
    tables: Mapping[int, pd.DataFrame],
    leads: Sequence[int],
    output_path: Path,
) -> None:
    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
        datetime_format="yyyy-mm",
    ) as writer:
        columns = [
            "IssueMonth",
            "TargetMonth",
            "Observed",
            "Predicted (Reference)",
            "Predicted (Seasonal Forecast)",
        ]

        for lead in leads:
            table = tables.get(
                lead,
                pd.DataFrame(columns=columns),
            )
            sheet_name = f"Lead{lead}"
            table.to_excel(
                writer,
                index=False,
                sheet_name=sheet_name,
            )

            worksheet = writer.sheets[sheet_name]
            worksheet.column_dimensions["A"].width = 15
            worksheet.column_dimensions["B"].width = 15
            worksheet.column_dimensions["C"].width = 14
            worksheet.column_dimensions["D"].width = 23
            worksheet.column_dimensions["E"].width = 30


def save_results(
    forecasts: pd.DataFrame,
    config: AppConfig,
    source_config_path: Path,
) -> None:
    output_directory = config.output.output_directory
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_tag = safe_filename(
        config.data.target
    )
    run_tag = safe_filename(
        config.output.run_name
    )
    prefix = f"{run_tag}_{target_tag}"

    forecasts = forecasts.copy()
    forecasts["IssueMonth"] = pd.to_datetime(
        forecasts["IssueMonth"]
    )
    forecasts["TargetMonth"] = pd.to_datetime(
        forecasts["TargetMonth"]
    )

    forecast_path = (
        output_directory
        / f"{prefix}_forecasts.csv"
    )
    forecasts.to_csv(
        forecast_path,
        index=False,
    )

    metrics_all = summarize_metrics(forecasts)
    metrics_all_path = (
        output_directory
        / f"{prefix}_metrics_all.csv"
    )
    metrics_all.to_csv(
        metrics_all_path,
        index=False,
    )

    year_tables: Dict[int, pd.DataFrame] = {}

    for year in sorted(
        forecasts["TargetMonth"].dt.year.unique()
    ):
        yearly_forecasts = forecasts[
            forecasts["TargetMonth"].dt.year
            == year
        ]
        yearly_metrics = summarize_metrics(
            yearly_forecasts
        )
        year_tables[int(year)] = yearly_metrics

        yearly_metrics.to_csv(
            output_directory
            / f"{prefix}_metrics_{year}.csv",
            index=False,
        )

    metrics_workbook_path = (
        output_directory
        / f"{prefix}_metrics.xlsx"
    )
    with pd.ExcelWriter(
        metrics_workbook_path,
        engine="openpyxl",
    ) as writer:
        metrics_all.to_excel(
            writer,
            sheet_name="All",
            index=False,
        )
        for year, table in year_tables.items():
            table.to_excel(
                writer,
                sheet_name=str(year),
                index=False,
            )

    if config.output.save_per_lead_workbook:
        per_lead_tables = build_per_lead_tables(
            forecasts
        )
        save_per_lead_workbook(
            tables=per_lead_tables,
            leads=config.forecast.leads,
            output_path=(
                output_directory
                / f"{prefix}_per_lead_predictions.xlsx"
            ),
        )

    copied_config_path = (
        output_directory
        / "configuration_used.json"
    )
    copied_config_path.write_text(
        source_config_path.read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    print(f"Saved forecasts: {forecast_path}")
    print(f"Saved metrics: {metrics_all_path}")
    print(f"Saved metrics workbook: {metrics_workbook_path}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    arguments = parse_arguments()
    config = load_configuration(
        arguments.config
    )

    set_random_seeds(
        config.model.random_seed
    )

    print(
        f"Target: {config.data.target}"
    )
    print(
        "History features: "
        f"{list(config.data.history_features)}"
    )
    print(
        "Future features: "
        f"{list(config.data.future_features)}"
    )
    print(
        "Forecast leads: "
        f"{list(config.forecast.leads)}"
    )
    print(
        "Lag steps: "
        f"{list(config.forecast.lag_steps)}"
    )
    print(
        "Retraining frequency: "
        f"{config.forecast.retrain_frequency}"
    )

    bundle = load_data(config)
    forecasts = run_forward_chain(
        bundle=bundle,
        config=config,
    )
    save_results(
        forecasts=forecasts,
        config=config,
        source_config_path=arguments.config,
    )

    print("\nMetrics for all target months:")
    print(
        summarize_metrics(
            forecasts
        ).to_string(index=False)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"LSTM forward-chain workflow failed: {error}",
            file=sys.stderr,
        )
        raise
