# backend/agents/data_agent.py
import numpy as np
import pandas as pd
import traceback

from utils.ml_preprocessing import detect_identifier_columns, normalize_raw_frame


class DataCleaningAgent:
    def run(self, df: pd.DataFrame, target_col: str = None) -> dict:
        try:
            cleaned = df.copy()
            original_shape = cleaned.shape
            original_missing = int(cleaned.isna().sum().sum())
            original_duplicates = int(cleaned.duplicated().sum())
            original_cells = max(int(cleaned.shape[0] * cleaned.shape[1]), 1)
            cleaning_log = []
            original_types = {col: str(dtype) for col, dtype in cleaned.dtypes.items()}

            cleaned, normalize_log, normalization_stats = normalize_raw_frame(cleaned, target_col)
            cleaning_log.extend(normalize_log)

            missing_pct_by_col = cleaned.isna().mean()
            drop_cols = [col for col in missing_pct_by_col[missing_pct_by_col > 0.7].index.tolist() if col != target_col]
            if drop_cols:
                cleaned = cleaned.drop(columns=drop_cols)
                cleaning_log.append(f"Dropped {len(drop_cols)} columns with more than 70% missing values: {', '.join(drop_cols)}")

            before_dupes = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            duplicates_removed = before_dupes - len(cleaned)
            if duplicates_removed:
                cleaning_log.append(f"Removed {duplicates_removed} exact duplicate rows.")

            identifier_columns = detect_identifier_columns(cleaned, target_col)
            if identifier_columns:
                for identifier in identifier_columns:
                    cleaning_log.append(
                        f"Dropped identifier column: {identifier.column} (unique ratio: {identifier.unique_ratio:.1f})"
                    )
                cleaned = cleaned.drop(columns=[identifier.column for identifier in identifier_columns])

            zero_variance_cols = [
                col for col in cleaned.columns if cleaned[col].nunique(dropna=False) <= 1 and col != target_col
            ]
            if zero_variance_cols:
                cleaned = cleaned.drop(columns=zero_variance_cols)
                cleaning_log.append(f"Dropped {len(zero_variance_cols)} zero-variance columns: {', '.join(zero_variance_cols)}")

            missing_fixed = max(original_missing - int(cleaned.isna().sum().sum()), 0)
            rows_removed = original_shape[0] - cleaned.shape[0]
            cols_removed = original_shape[1] - cleaned.shape[1]
            missing_pct = original_missing / original_cells * 100
            duplicate_pct = original_duplicates / max(original_shape[0], 1) * 100
            cols_removed_pct = cols_removed / max(original_shape[1], 1) * 100
            data_quality_score = float(np.clip(100 - (missing_pct * 0.5 + duplicate_pct * 0.3 + cols_removed_pct * 0.2), 0, 100))

            if not cleaning_log:
                cleaning_log.append("Dataset already met cleaning requirements; no structural changes were needed.")

            # FINAL SAFETY PASS: Ensure no string[python] or pd.NA leak into later pipeline steps
            for col in cleaned.select_dtypes(include=["string"]).columns:
                cleaned[col] = cleaned[col].astype(object)
            for col in cleaned.select_dtypes(include=["boolean"]).columns:
                cleaned[col] = cleaned[col].astype(bool)
            cleaned = cleaned.replace({pd.NA: np.nan})

            return {
                "status": "success",
                "cleaned_df": cleaned,
                "original_shape": tuple(original_shape),
                "cleaned_shape": tuple(cleaned.shape),
                "rows_removed": int(rows_removed),
                "cols_removed": int(cols_removed),
                "missing_fixed": int(missing_fixed),
                "duplicates_removed": int(duplicates_removed),
                "data_quality_score": round(data_quality_score, 2),
                "cleaning_log": cleaning_log,
                "identifier_columns": [
                    {
                        "column": identifier.column,
                        "unique_ratio": round(identifier.unique_ratio, 4),
                        "reason": identifier.reason,
                    }
                    for identifier in identifier_columns
                ],
                "normalization_stats": normalization_stats,
                "column_types": {
                    col: {"original": original_types.get(col, "unknown"), "cleaned": str(dtype)}
                    for col, dtype in cleaned.dtypes.items()
                },
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
