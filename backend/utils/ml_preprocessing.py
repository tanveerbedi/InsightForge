import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler, StandardScaler


ID_NAME_PATTERNS = ("id", "customerid", "userid", "uuid", "transactionid", "accountid", "sessionid")


@dataclass
class IdentifierColumn:
    column: str
    unique_ratio: float
    reason: str


def detect_identifier_columns(df: pd.DataFrame, target_col: str | None = None, unique_ratio_threshold: float = 0.9) -> list[IdentifierColumn]:
    """Detect columns that behave like row identifiers and should not be modeled."""
    identifiers: list[IdentifierColumn] = []
    rows = max(len(df), 1)
    target_norm = _norm(target_col) if target_col else None

    for col in df.columns:
        if target_col and col == target_col:
            continue
        series = df[col]
        unique_ratio = float(series.nunique(dropna=False) / rows)
        name = _norm(col)
        looks_like_name = any(pattern in name for pattern in ID_NAME_PATTERNS)
        near_unique_text = not pd.api.types.is_numeric_dtype(series) and unique_ratio >= unique_ratio_threshold
        monotonic_numeric = pd.api.types.is_numeric_dtype(series) and unique_ratio >= unique_ratio_threshold and (
            series.dropna().is_monotonic_increasing or series.dropna().is_monotonic_decreasing
        )
        sequential_text = near_unique_text and _has_sequential_tokens(series)

        if target_norm and name == target_norm:
            continue
        if looks_like_name and unique_ratio > 0.5:
            identifiers.append(IdentifierColumn(col, unique_ratio, "identifier-like column name"))
        elif unique_ratio >= 0.98 and (near_unique_text or monotonic_numeric or sequential_text):
            identifiers.append(IdentifierColumn(col, unique_ratio, "near-unique row identifier"))
        elif unique_ratio >= unique_ratio_threshold and looks_like_name:
            identifiers.append(IdentifierColumn(col, unique_ratio, "high-cardinality identifier"))

    return identifiers


def normalize_raw_frame(df: pd.DataFrame, target_col: str | None = None) -> tuple[pd.DataFrame, list[str], dict]:
    """Apply split-safe structural cleanup only; modeling transforms happen later."""
    cleaned = df.copy()
    logs: list[str] = []
    stats = {"coerced_numeric": [], "normalized_categoricals": []}

    # Replace inf with nan
    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in cleaned.select_dtypes(include=["object", "string", "category"]).columns:
        # Attempt to parse datetime
        try:
            parsed = pd.to_datetime(cleaned[col], errors='ignore')
            if pd.api.types.is_datetime64_any_dtype(parsed):
                cleaned[f"{col}_year"] = parsed.dt.year
                cleaned[f"{col}_month"] = parsed.dt.month
                cleaned[f"{col}_day"] = parsed.dt.day
                cleaned[f"{col}_dayofweek"] = parsed.dt.dayofweek
                cleaned[f"{col}_hour"] = parsed.dt.hour
                cleaned = cleaned.drop(columns=[col])
                logs.append(f"Parsed datetime column '{col}' into year/month/day/hour features.")
                continue
        except Exception:
            pass

        # If not datetime, treat as categorical
        cleaned[col] = cleaned[col].astype("string").str.strip()
        cleaned[col] = cleaned[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        
        # Limit high-cardinality categories to top 50, rest as "Other"
        n_unique = cleaned[col].nunique(dropna=True)
        if n_unique > 50:
            top_cats = cleaned[col].value_counts().nlargest(50).index
            cleaned.loc[~cleaned[col].isin(top_cats) & cleaned[col].notna(), col] = "Other"
            logs.append(f"Limited high-cardinality column '{col}' to top 50 categories.")

        # FIX: Convert back to object so missing values are handled as np.nan
        # scikit-learn's SimpleImputer crashes on pd.NA (TypeError: boolean value of NA is ambiguous)
        cleaned[col] = cleaned[col].astype(object)
        cleaned[col] = cleaned[col].fillna(np.nan)
        stats["normalized_categoricals"].append(col)

    for col in cleaned.columns:
        if col == target_col or pd.api.types.is_numeric_dtype(cleaned[col]):
            continue
        coerced = pd.to_numeric(cleaned[col], errors="coerce")
        original_missing = cleaned[col].isna()
        valid_ratio = float((coerced.notna() | original_missing).mean())
        if valid_ratio >= 0.95:
            cleaned[col] = coerced
            stats["coerced_numeric"].append(col)
            logs.append(f"Corrected numeric-like column '{col}' to numeric dtype.")

    return cleaned, logs, stats


def build_preprocessor(
    X: pd.DataFrame,
    model_family: str,
    ordinal_columns: Iterable[str] | None = None,
    rare_category_frequency: float = 0.01,
) -> ColumnTransformer:
    # FIX: Safety sanitization before scikit-learn transformers
    # Ensure no string[python] dtypes or pd.NA values exist
    for col in X.select_dtypes(include=["string"]).columns:
        X[col] = X[col].astype(object)
    X.replace({pd.NA: np.nan}, inplace=True)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    ordinal_columns = [c for c in (ordinal_columns or []) if c in categorical_cols]
    nominal_cols = [c for c in categorical_cols if c not in ordinal_columns]

    scaler = "passthrough"
    if model_family in {"linear", "svm", "knn"}:
        scaler = StandardScaler()
    elif model_family == "robust":
        scaler = RobustScaler()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if _needs_power_transform(X[numeric_cols]) and model_family in {"linear", "svm", "knn"}:
        numeric_steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=False)))
    if scaler != "passthrough":
        numeric_steps.append(("scaler", scaler))

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_cols))
    if ordinal_columns:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                ordinal_columns,
            )
        )
    if nominal_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=rare_category_frequency,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                nominal_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def get_feature_names(preprocessor: ColumnTransformer, fallback: list[str]) -> list[str]:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return fallback


def target_positive_label(classes: list[str]) -> int:
    normalized = [_norm(c) for c in classes]
    for token in ("yes", "true", "1", "churn", "positive"):
        if token in normalized:
            return normalized.index(token)
    return 1 if len(classes) > 1 else 0


def _norm(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _has_sequential_tokens(series: pd.Series) -> bool:
    values = series.dropna().astype(str).head(200)
    if values.empty:
        return False
    numeric_suffixes = values.str.extract(r"(\d+)$")[0].dropna()
    return len(numeric_suffixes) >= max(20, len(values) * 0.5)


def _needs_power_transform(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    try:
        skewed = df.skew(numeric_only=True).abs()
        return bool((skewed > 2).any())
    except Exception:
        return False
