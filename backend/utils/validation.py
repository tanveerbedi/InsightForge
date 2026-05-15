# backend/utils/validation.py
import pandas as pd
import numpy as np

def validate_dataset(df: pd.DataFrame, target_column: str = None) -> dict:
    """
    Perform pre-flight checks on the dataset to ensure it's valid for the ML pipeline.
    """
    if df is None or df.empty:
        return {"valid": False, "errors": ["Dataset is completely empty."], "warnings": [], "task_type": "unknown"}

    n_rows, n_cols = df.shape
    errors = []
    warnings = []
    task_type = "unknown"

    if n_rows < 10:
        errors.append(f"Dataset must have at least 10 rows for training (found {n_rows}).")

    if n_cols < 2:
        errors.append(f"Dataset must have at least 2 columns to extract features and a target (found {n_cols}).")

    if target_column:
        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not found in the dataset.")
        else:
            series = df[target_column]
            n_unique = int(series.nunique(dropna=True))
            null_pct = float(series.isna().mean())

            if null_pct > 0.5:
                errors.append(f"Target column '{target_column}' has {null_pct:.0%} missing values.")
            
            if n_unique < 2:
                errors.append(f"Target column '{target_column}' has only {n_unique} unique value(s). Must have at least 2.")
            
            if n_unique == n_rows and n_rows > 0:
                errors.append(f"Target column '{target_column}' has a unique value for every row (acts like an ID).")

            # Task detection
            if pd.api.types.is_numeric_dtype(series) and n_unique > 10:
                task_type = "regression"
            else:
                task_type = "classification"
    else:
        # We don't have a target column yet. The auto-detection will happen later.
        pass

    valid = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "task_type": task_type
    }
