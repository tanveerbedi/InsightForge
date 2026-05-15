# backend/agents/eda_agent.py
import base64
import traceback
from io import BytesIO

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from utils.ml_preprocessing import detect_identifier_columns


class EDAAgent:
    def run(self, df: pd.DataFrame, target_col: str = None, problem_type: str = None) -> dict:
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            numeric_features = numeric_df.drop(columns=[target_col], errors="ignore")
            missing_per_column = {col: int(val) for col, val in df.isna().sum().items()}
            recommended_target = target_col or self._recommend_target(df)
            class_balance = {}

            if problem_type == "classification" and recommended_target in df.columns:
                counts = df[recommended_target].value_counts(dropna=False)
                total = max(int(counts.sum()), 1)
                class_balance = {
                    str(label): {"count": int(count), "percentage": round(float(count) / total * 100, 2)}
                    for label, count in counts.items()
                }

            feature_distributions = {}
            for col in numeric_features.columns:
                values = numeric_df[col].dropna()
                if values.empty:
                    continue
                counts, bins = np.histogram(values, bins=20)
                feature_distributions[col] = {
                    "counts": counts.tolist(),
                    "bins": bins.tolist(),
                    "skewness": float(values.skew()) if len(values) > 2 else 0.0,
                    "kurtosis": float(values.kurtosis()) if len(values) > 3 else 0.0,
                }

            identifier_risks = detect_identifier_columns(df, target_col)
            cardinality = self._cardinality(df, target_col)
            outliers = self._outliers(numeric_features)
            mutual_information = self._mutual_information(df, recommended_target, problem_type)
            target_distributions = self._target_distributions(df, recommended_target, problem_type)
            insights = self._build_insights(df, numeric_features, class_balance, identifier_risks, cardinality, outliers, mutual_information, target_col)
            charts = {
                "correlation_matrix": self._chart_correlation(numeric_features),
                "boxplots": self._chart_boxplots(numeric_features),
                "missing_heatmap": self._chart_missing(df),
                "distributions": {
                    col: self._chart_distribution(numeric_features[col], col)
                    for col in numeric_features.columns[:5]
                },
            }

            return {
                "status": "success",
                "problem_type": problem_type,
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_per_column": missing_per_column,
                "describe": df.describe(include="all").astype(object).where(pd.notna(df.describe(include="all")), "").to_dict(),
                "correlation": numeric_df.corr(numeric_only=True).fillna(0).to_dict(),
                "cardinality": cardinality,
                "outliers": outliers,
                "mutual_information": mutual_information,
                "target_distributions": target_distributions,
                "identifier_risks": [
                    {"column": item.column, "unique_ratio": round(item.unique_ratio, 4), "reason": item.reason}
                    for item in identifier_risks
                ],
                "class_balance": class_balance,
                "feature_distributions": feature_distributions,
                "recommended_target": recommended_target,
                "insights": insights,
                "charts": charts,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}

    def _recommend_target(self, df):
        if df.empty:
            return None
        scores = {}
        rows = max(len(df), 1)
        for col in df.columns:
            unique_ratio = df[col].nunique(dropna=False) / rows
            if pd.api.types.is_numeric_dtype(df[col]):
                scores[col] = float(df[col].var() or 0) + unique_ratio
            else:
                scores[col] = unique_ratio * 10
        return max(scores, key=scores.get)

    def _build_insights(self, df, numeric_df, class_balance, identifier_risks, cardinality, outliers, mutual_information, target_col):
        insights = []
        for item in identifier_risks:
            insights.append(
                {
                    "severity": "HIGH",
                    "topic": "Identifier leakage risk",
                    "description": f"'{item.column}' looks like an identifier ({item.unique_ratio:.1%} unique) and should be excluded from modeling.",
                }
            )
        for col, pct in df.isna().mean().items():
            if pct > 0.3:
                insights.append(
                    {
                        "severity": "HIGH",
                        "topic": "High missingness",
                        "description": f"Column '{col}' has {pct * 100:.1f}% missing values.",
                    }
                )
        if class_balance:
            counts = [v["count"] for v in class_balance.values()]
            if counts and min(counts) > 0 and max(counts) / min(counts) > 5:
                insights.append(
                    {
                        "severity": "HIGH",
                        "topic": "Class imbalance",
                        "description": f"The largest class is {max(counts) / min(counts):.1f}x the smallest class.",
                    }
                )
        corr = numeric_df.corr(numeric_only=True).abs()
        for i, col in enumerate(corr.columns):
            for other in corr.columns[i + 1 :]:
                value = corr.loc[col, other]
                if pd.notna(value) and value > 0.9:
                    insights.append(
                        {
                            "severity": "MEDIUM",
                            "topic": "High correlation",
                            "description": f"'{col}' and '{other}' are strongly correlated ({value:.2f}).",
                        }
                    )
        for col in numeric_df.columns:
            if numeric_df[col].nunique(dropna=False) <= 2:
                insights.append(
                    {
                        "severity": "MEDIUM",
                        "topic": "Low variance",
                        "description": f"'{col}' has very few distinct values and may contribute little signal.",
                    }
                )
            skewness = numeric_df[col].dropna().skew() if len(numeric_df[col].dropna()) > 2 else 0
            if pd.notna(skewness) and abs(skewness) > 2:
                insights.append(
                    {
                        "severity": "LOW",
                        "topic": "Skewed distribution",
                        "description": f"'{col}' is highly skewed (skewness {skewness:.2f}).",
                    }
                )
            kurtosis = numeric_df[col].dropna().kurtosis() if len(numeric_df[col].dropna()) > 3 else 0
            if pd.notna(kurtosis) and abs(kurtosis) > 5:
                insights.append(
                    {
                        "severity": "LOW",
                        "topic": "Heavy-tailed distribution",
                        "description": f"'{col}' has high kurtosis ({kurtosis:.2f}); robust scaling or transformation may help linear models.",
                    }
                )
        for item in cardinality[:10]:
            if item["unique_ratio"] > 0.5 and item["column"] != target_col:
                insights.append(
                    {
                        "severity": "MEDIUM",
                        "topic": "High cardinality",
                        "description": f"'{item['column']}' has {item['unique_count']} unique values; rare category grouping is recommended.",
                    }
                )
        for col, payload in outliers.items():
            if payload["outlier_pct"] > 5:
                insights.append(
                    {
                        "severity": "MEDIUM",
                        "topic": "Outlier concentration",
                        "description": f"'{col}' has {payload['outlier_pct']:.1f}% IQR outliers.",
                    }
                )
        top_mi = mutual_information[:1]
        if top_mi and top_mi[0]["score"] < 0.001:
            insights.append(
                {
                    "severity": "MEDIUM",
                    "topic": "Weak feature signal",
                    "description": "Mutual information did not find a strong individual predictor; expect model performance to depend on interactions.",
                }
            )
        if not insights:
            insights.append(
                {
                    "severity": "LOW",
                    "topic": "Stable profile",
                    "description": "No severe data quality or distribution risks were detected in the cleaned dataset.",
                }
            )
        return insights[:20]

    def _cardinality(self, df, target_col):
        rows = max(len(df), 1)
        result = []
        for col in df.columns:
            if col == target_col:
                continue
            unique = int(df[col].nunique(dropna=False))
            probs = df[col].value_counts(normalize=True, dropna=False)
            entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
            result.append({"column": col, "unique_count": unique, "unique_ratio": round(unique / rows, 4), "entropy": round(entropy, 4)})
        return sorted(result, key=lambda x: x["unique_ratio"], reverse=True)

    def _outliers(self, numeric_df):
        result = {}
        for col in numeric_df.columns:
            values = numeric_df[col].dropna()
            if len(values) < 5:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
            result[col] = {"count": int(mask.sum()), "outlier_pct": round(float(mask.mean() * 100), 2)}
        return result

    def _mutual_information(self, df, target_col, problem_type):
        if not target_col or target_col not in df.columns:
            return []
        try:
            X = df.drop(columns=[target_col]).copy()
            y = df[target_col]
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = pd.factorize(X[col].astype(str), sort=True)[0]
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else 0)
            if not pd.api.types.is_numeric_dtype(y):
                y = pd.factorize(y.astype(str), sort=True)[0]
            func = mutual_info_classif if problem_type == "classification" else mutual_info_regression
            scores = func(X, y, random_state=42)
            return sorted(
                [{"feature": str(col), "score": float(score)} for col, score in zip(X.columns, scores)],
                key=lambda x: x["score"],
                reverse=True,
            )[:20]
        except Exception:
            return []

    def _target_distributions(self, df, target_col, problem_type):
        if problem_type != "classification" or not target_col or target_col not in df.columns:
            return {}
        result = {}
        for col in df.columns:
            if col == target_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                grouped = df.groupby(target_col)[col].agg(["mean", "median", "std"]).fillna(0)
                result[col] = grouped.to_dict(orient="index")
            elif df[col].nunique(dropna=False) <= 20:
                table = pd.crosstab(df[col], df[target_col], normalize="index").round(4)
                result[col] = table.to_dict(orient="index")
        return dict(list(result.items())[:20])

    def _encode_plot(self, plot_fn):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            plot_fn(plt, sns)
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            plt.close()
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception:
            return {"type": "unavailable", "reason": "Chart generation requires kaleido or a plotting backend."}

    def _chart_correlation(self, numeric_df):
        if numeric_df.shape[1] < 2:
            return None
        return self._encode_plot(
            lambda plt, sns: (
                plt.figure(figsize=(8, 6)),
                sns.heatmap(numeric_df.corr(numeric_only=True), cmap="viridis", annot=False),
                plt.title("Correlation Matrix"),
            )
        )

    def _chart_boxplots(self, numeric_df):
        if numeric_df.empty:
            return None
        cols = numeric_df.columns[:10]
        return self._encode_plot(
            lambda plt, sns: (
                plt.figure(figsize=(10, 5)),
                sns.boxplot(data=numeric_df[cols], orient="h"),
                plt.title("Numeric Feature Boxplots"),
            )
        )

    def _chart_missing(self, df):
        missing = df.isna().sum()
        if int(missing.sum()) == 0:
            missing = pd.Series({col: 0 for col in df.columns[:10]})
        return self._encode_plot(
            lambda plt, sns: (
                plt.figure(figsize=(9, 4)),
                sns.barplot(x=missing.index[:20], y=missing.values[:20]),
                plt.ylim(bottom=0),
                plt.xticks(rotation=45, ha="right"),
                plt.title("Missing Values by Column"),
            )
        )

    def _chart_distribution(self, series, col):
        values = series.dropna()
        if values.empty:
            return None
        return self._encode_plot(
            lambda plt, sns: (
                plt.figure(figsize=(7, 4)),
                sns.histplot(values, kde=True, bins=20),
                plt.title(f"Distribution: {col}"),
            )
        )
