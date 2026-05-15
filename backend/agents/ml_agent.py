import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from utils.ml_preprocessing import build_preprocessor, get_feature_names, target_positive_label
from utils.serializer import make_serializable
import logging

# Configure model training logger
logger = logging.getLogger("ml_agent")
logger.setLevel(logging.INFO)
log_path = Path(".storage/logs/model_training.log")
log_path.parent.mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(log_path)
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(fh)

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


RANDOM_STATE = 42


class MLAgent:
    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        problem_type: str,
        selected_models: list = None,
        user_params: dict = None,
        fast_mode: bool = False,
        progress_callback=None,
        run_id: str = "latest",
    ) -> dict:
        try:
            if target_col not in df.columns:
                return {"status": "error", "error": f"Target column '{target_col}' not found."}
            if df[target_col].isna().any():
                df = df.dropna(subset=[target_col]).copy()

            # FIX: Pre-training dtype sanitization
            for col in df.select_dtypes(include=["string"]).columns:
                df[col] = df[col].astype(object)
            for col in df.select_dtypes(include=["boolean"]).columns:
                df[col] = df[col].astype(bool)
            df.replace({pd.NA: np.nan}, inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            X = df.drop(columns=[target_col])
            y_raw = df[target_col]
            target_encoder = None
            target_classes = []
            positive_label = None
            if problem_type == "classification":
                target_encoder = LabelEncoder()
                y = pd.Series(target_encoder.fit_transform(y_raw.astype(str)), index=y_raw.index)
                target_classes = [str(c) for c in target_encoder.classes_]
                positive_label = target_positive_label(target_classes)
            else:
                y = pd.to_numeric(y_raw, errors="coerce")
                valid = y.notna()
                X = X.loc[valid]
                y = y.loc[valid]

            X = X.loc[y.index]
            if len(y) < 10:
                return {"status": "error", "error": f"Too few samples ({len(y)}) after cleaning. Need at least 10 rows for modeling."}
            if X.empty or X.shape[1] == 0:
                return {"status": "error", "error": "No valid features remaining after preprocessing. Check that your dataset has feature columns beyond the target."}
            if y.nunique(dropna=True) < 2:
                return {"status": "error", "error": "Target column has less than 2 unique values after cleaning."}

            stratify = None
            if problem_type == "classification" and pd.Series(y).nunique() > 1:
                class_counts = pd.Series(y).value_counts()
                if int(class_counts.min()) >= 2:
                    stratify = y
                else:
                    logger.warning("Stratification disabled because at least one class has fewer than 2 samples.")
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
                )
            except ValueError as e:
                logger.warning(f"Stratified split failed: {e}. Falling back to unstratified split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_STATE
                )

            imbalance = self._imbalance_diagnostics(y_train, problem_type)
            registry, unavailable = self._registry(problem_type, fast_mode, imbalance)
            defaults = (
                ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier"]
                if problem_type == "classification"
                else ["LinearRegression", "RandomForestRegressor", "XGBRegressor"]
            )
            selected_models = selected_models or defaults

            jobs, all_models = [], []
            for name in selected_models:
                if name in unavailable:
                    all_models.append(self._failed(name, unavailable[name]))
                    continue
                if name not in registry:
                    all_models.append(self._failed(name, "Model is not available in the registry."))
                    continue
                estimator, family = registry[name]
                params = (user_params or {}).get(name, {})
                if params:
                    estimator.set_params(**params)
                jobs.append((name, estimator, family))

            try:
                trained = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(self._train_one)(
                        name,
                        estimator,
                        family,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        problem_type,
                        imbalance,
                        positive_label,
                    )
                    for name, estimator, family in jobs
                )
            except PermissionError:
                trained = [
                    self._train_one(name, estimator, family, X_train, X_test, y_train, y_test, problem_type, imbalance, positive_label)
                    for name, estimator, family in jobs
                ]
            all_models.extend(trained)
            for result in trained:
                if progress_callback:
                    progress_callback(result.get("name", "model"), "done", f"Finished {result.get('name')}")

            metric = "churn_recall" if problem_type == "classification" and positive_label is not None else "r2"
            ok = [m for m in all_models if not m.get("error") and m.get("metrics")]
            if not ok:
                diagnostics = {
                    "task_type": problem_type,
                    "n_rows": int(len(X_train) + len(X_test)),
                    "n_features": int(X_train.shape[1]),
                    "feature_names": X_train.columns.tolist()[:50],  # up to 50 for brevity
                    "target_column": target_col,
                    "target_unique_values": int(y.nunique(dropna=True)),
                }
                model_failures = [{"name": m["name"], "error": m["error"]} for m in all_models if m.get("error")]
                return make_serializable({
                    "status": "error", 
                    "error": "All models failed to train. No models trained successfully.", 
                    "all_models": all_models,
                    "model_failures": model_failures,
                    "diagnostics": diagnostics,
                    "suggestions": [
                        "Check if the dataset contains unsupported characters or extreme outliers.",
                        "Verify that numeric columns do not contain mixed string types, NaN, or infinite values.",
                        "If using a very large dataset, try reducing the number of categorical features."
                    ]
                })
            ok.sort(
                key=lambda m: (
                    (m.get("tuned_metrics") or m.get("metrics")).get(metric, -np.inf),
                    (m.get("tuned_metrics") or m.get("metrics")).get("f1_macro", -np.inf),
                    (m.get("tuned_metrics") or m.get("metrics")).get("f1_weighted", -np.inf),
                ),
                reverse=True,
            )
            for i, item in enumerate(ok, 1):
                item["rank"] = i

            self._tune(ok[:3], X_train, X_test, y_train, y_test, problem_type, fast_mode, positive_label, metric)
            ok.sort(
                key=lambda m: (
                    (m.get("tuned_metrics") or m.get("metrics")).get(metric, -np.inf),
                    (m.get("tuned_metrics") or m.get("metrics")).get("f1_macro", -np.inf),
                    (m.get("tuned_metrics") or m.get("metrics")).get("f1_weighted", -np.inf),
                ),
                reverse=True,
            )
            for i, item in enumerate(ok, 1):
                item["rank"] = i

            best = ok[0]
            model_dir = Path("./.storage/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{run_id}.pkl"
            joblib.dump(best["fitted_model"], model_path)

            best_metrics = best.get("tuned_metrics") or best["metrics"]
            second = ok[1] if len(ok) > 1 else None
            second_score = ((second.get("tuned_metrics") or second.get("metrics")).get(metric, 0) if second else 0)
            delta = float(best_metrics.get(metric, 0) - second_score)
            params = best.get("tuned_params") or best["fitted_model"].named_steps["model"].get_params()
            why_best = (
                f"{best['name']} was selected on churn-aware validation: {metric}={best_metrics.get(metric, 0):.4f}, "
                f"ahead of {second['name'] if second else 'the baseline'} by +{delta:.4f}. "
                f"Weighted F1={best_metrics.get('f1_weighted', 0):.4f}, macro F1={best_metrics.get('f1_macro', 0):.4f}."
            )
            returned_models = [{k: v for k, v in m.items() if k not in {"fitted_model", "y_pred", "y_score"}} for m in all_models]

            y_pred = best.get("y_pred", [])
            y_score = best.get("y_score", [])
            result = {
                "status": "success",
                "problem_type": problem_type,
                "all_models": returned_models,
                "best_model_name": best["name"],
                "best_params": params,
                "best_metrics": best_metrics,
                "why_best": why_best,
                "feature_names": best.get("feature_names", X.columns.tolist()),
                "raw_feature_names": X.columns.tolist(),
                "preprocessing_notes": self._notes(problem_type, imbalance),
                "class_imbalance_handled": imbalance["strategy"] != "none",
                "smote_applied": "SMOTE" in imbalance["strategy"],
                "imbalance_ratio": round(float(imbalance["ratio"]), 4),
                "imbalance_diagnostics": imbalance,
                "train_size": int(len(X_train)),
                "resampled_train_size": int(best.get("resampled_train_size") or len(X_train)),
                "test_size": int(len(X_test)),
                "scaler_used": best.get("scaler_used", "none"),
                "model_path": str(model_path),
                "target_classes": target_classes,
                "positive_class": target_classes[positive_label] if positive_label is not None and target_classes else None,
                "positive_label": positive_label,
                "y_test": pd.Series(y_test).tolist(),
                "y_pred": pd.Series(y_pred).tolist(),
                "y_score": pd.Series(y_score).tolist(),
                "holdout_y_true": pd.Series(y_test).tolist(),
                "holdout_y_pred": pd.Series(y_pred).tolist(),
                "holdout_y_score": pd.Series(y_score).tolist(),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist() if problem_type == "classification" else [],
                "normalized_confusion_matrix": self._normalized_confusion_matrix(y_test, y_pred) if problem_type == "classification" else [],
                "threshold_optimization": best_metrics.get("threshold_optimization", {}),
            }
            return make_serializable(result)
        except Exception as exc:
            return make_serializable({"status": "error", "error": str(exc), "traceback": traceback.format_exc()})

    def _registry(self, problem_type, fast_mode, imbalance):
        unavailable = {}
        if problem_type == "classification":
            ratio = max(float(imbalance.get("ratio", 1.0)), 1.0)
            registry = {
                "LogisticRegression": (
                    LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight="balanced"),
                    "linear",
                ),
                "DecisionTreeClassifier": (
                    DecisionTreeClassifier(max_depth=None, class_weight="balanced", random_state=RANDOM_STATE),
                    "tree",
                ),
                "RandomForestClassifier": (
                    RandomForestClassifier(
                        n_estimators=250 if not fast_mode else 100,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                    "tree",
                ),
                "ExtraTreesClassifier": (
                    ExtraTreesClassifier(
                        n_estimators=250 if not fast_mode else 100,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                    "tree",
                ),
                "GradientBoostingClassifier": (
                    GradientBoostingClassifier(n_estimators=200 if not fast_mode else 80, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
                    "boosting",
                ),
                "HistGradientBoostingClassifier": (
                    HistGradientBoostingClassifier(max_iter=200 if not fast_mode else 80, learning_rate=0.05, random_state=RANDOM_STATE),
                    "boosting",
                ),
                "SVC": (SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced"), "svm"),
                "KNeighborsClassifier": (KNeighborsClassifier(n_neighbors=5, weights="distance"), "knn"),
            }
            if XGB_AVAILABLE:
                registry["XGBClassifier"] = (
                    XGBClassifier(
                        n_estimators=300 if not fast_mode else 100,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        scale_pos_weight=ratio,
                        verbosity=0,
                        random_state=RANDOM_STATE,
                    ),
                    "boosting",
                )
            else:
                unavailable["XGBClassifier"] = "xgboost is not installed."
            if LGBM_AVAILABLE:
                registry["LGBMClassifier"] = (
                    LGBMClassifier(
                        n_estimators=300 if not fast_mode else 100,
                        learning_rate=0.05,
                        num_leaves=31,
                        class_weight="balanced",
                        verbose=-1,
                        random_state=RANDOM_STATE,
                    ),
                    "boosting",
                )
            else:
                unavailable["LGBMClassifier"] = "lightgbm is not installed."
            if CATBOOST_AVAILABLE:
                registry["CatBoostClassifier"] = (
                    CatBoostClassifier(
                        iterations=300 if not fast_mode else 100,
                        learning_rate=0.05,
                        depth=6,
                        auto_class_weights="Balanced",
                        verbose=0,
                        random_state=RANDOM_STATE,
                    ),
                    "boosting",
                )
            else:
                unavailable["CatBoostClassifier"] = "catboost is not installed."
            return registry, unavailable

        registry = {
            "LinearRegression": (LinearRegression(), "linear"),
            "Ridge": (Ridge(alpha=1.0), "linear"),
            "Lasso": (Lasso(alpha=0.1), "linear"),
            "RandomForestRegressor": (RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE), "tree"),
            "GradientBoostingRegressor": (GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE), "boosting"),
            "SVR": (SVR(C=1.0, kernel="rbf"), "svm"),
        }
        if XGB_AVAILABLE:
            registry["XGBRegressor"] = (
                XGBRegressor(n_estimators=300 if not fast_mode else 100, learning_rate=0.05, max_depth=6, verbosity=0, random_state=RANDOM_STATE),
                "boosting",
            )
        else:
            unavailable["XGBRegressor"] = "xgboost is not installed."
        if LGBM_AVAILABLE:
            registry["LGBMRegressor"] = (
                LGBMRegressor(n_estimators=300 if not fast_mode else 100, learning_rate=0.05, verbose=-1, random_state=RANDOM_STATE),
                "boosting",
            )
        else:
            unavailable["LGBMRegressor"] = "lightgbm is not installed."
        if CATBOOST_AVAILABLE:
            registry["CatBoostRegressor"] = (
                CatBoostRegressor(iterations=300 if not fast_mode else 100, learning_rate=0.05, verbose=0, random_state=RANDOM_STATE),
                "boosting",
            )
        else:
            unavailable["CatBoostRegressor"] = "catboost is not installed."
        return registry, unavailable

    def _train_one(self, name, estimator, family, X_train, X_test, y_train, y_test, problem_type, imbalance, positive_label):
        started = time.time()
        try:
            preprocessor = build_preprocessor(X_train, family)
            steps = [("preprocessor", preprocessor), ("variance", VarianceThreshold())]
            sampler = self._sampler_for_model(name, family, imbalance, problem_type)
            if sampler is not None:
                steps.append(("sampler", sampler))
                pipe = ImbPipeline(steps + [("model", estimator)])
            else:
                pipe = SklearnPipeline(steps + [("model", estimator)])

            pipe.fit(X_train, y_train)
            y_score = self._scores(pipe, X_test, problem_type, positive_label)
            y_pred = self._predict_with_threshold(pipe, X_test, y_score, problem_type, positive_label)
            feature_names = get_feature_names(pipe.named_steps["preprocessor"], X_train.columns.tolist())
            metrics = self._metrics(pipe, X_test, y_test, y_pred, y_score, problem_type, positive_label)
            logger.info(f"Successfully trained {name}")
            return {
                "name": name,
                "display_name": name.replace("Classifier", "").replace("Regressor", ""),
                "metrics": metrics,
                "training_time_sec": round(time.time() - started, 3),
                "tuned": False,
                "tuned_params": {},
                "tuned_metrics": {},
                "rank": None,
                "error": None,
                "fitted_model": pipe,
                "scaler_used": self._scaler_name(family),
                "imbalance_strategy": sampler.__class__.__name__ if sampler is not None else self._weight_strategy(name, imbalance),
                "feature_names": feature_names,
                "y_pred": y_pred,
                "y_score": y_score,
            }
        except Exception as exc:
            error_msg = str(exc)
            logger.error(f"Failed to train {name}:\n{traceback.format_exc()}")
            return self._failed(name, error_msg, time.time() - started)

    def _metrics(self, model, X_test, y_test, y_pred, y_score, problem_type, positive_label):
        if problem_type == "classification":
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            labels = sorted(pd.Series(y_test).unique().tolist())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            tn = cm[0, 0] if cm.shape == (2, 2) else 0
            fp = cm[0, 1] if cm.shape == (2, 2) else 0
            specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "specificity": specificity,
                "confusion_matrix": cm.tolist(),
                "normalized_confusion_matrix": self._normalized_confusion_matrix(y_test, y_pred),
                "classification_report": report,
            }
            if positive_label is not None and str(positive_label) in report:
                metrics["churn_recall"] = report[str(positive_label)]["recall"]
                metrics["churn_precision"] = report[str(positive_label)]["precision"]
                metrics["churn_f1"] = report[str(positive_label)]["f1-score"]
            if len(np.unique(y_test)) == 2 and len(y_score):
                try:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_score)
                    metrics["pr_auc"] = average_precision_score(y_test, y_score)
                    metrics["threshold_optimization"] = self._threshold_optimization(y_test, y_score, positive_label)
                except Exception:
                    metrics["roc_auc"] = None
                    metrics["pr_auc"] = None
            return metrics

        y_test_arr = np.asarray(y_test, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        mse = mean_squared_error(y_test_arr, y_pred_arr)
        denom = np.where(y_test_arr == 0, np.nan, y_test_arr)
        mape = np.nanmean(np.abs((y_test_arr - y_pred_arr) / denom)) * 100
        return {
            "mae": mean_absolute_error(y_test_arr, y_pred_arr),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": r2_score(y_test_arr, y_pred_arr),
            "mape": None if np.isnan(mape) else float(mape),
        }

    def _tune(self, top_models, X_train, X_test, y_train, y_test, problem_type, fast_mode, positive_label, scoring_metric):
        scoring = "recall" if scoring_metric == "churn_recall" else ("r2" if problem_type == "regression" else "f1_macro")
        if problem_type == "classification":
            class_counts = pd.Series(y_train).value_counts()
            max_splits = int(class_counts.min()) if not class_counts.empty else 0
            n_splits = min(3 if fast_mode else 5, max_splits)
            if n_splits < 2:
                return
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        else:
            n_splits = min(3, len(y_train))
            if n_splits < 2:
                return
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for item in top_models:
            grid = self._grid(item["name"])
            if not grid:
                continue
            try:
                search = RandomizedSearchCV(
                    item["fitted_model"],
                    {f"model__{key}": value for key, value in grid.items()},
                    n_iter=8 if fast_mode else 15,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                    error_score="raise",
                )
                search.fit(X_train, y_train)
                scores = self._scores(search.best_estimator_, X_test, problem_type, positive_label)
                pred = self._predict_with_threshold(search.best_estimator_, X_test, scores, problem_type, positive_label)
                item.update(
                    {
                        "tuned": True,
                        "tuned_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                        "tuned_metrics": self._metrics(search.best_estimator_, X_test, y_test, pred, scores, problem_type, positive_label),
                        "fitted_model": search.best_estimator_,
                        "y_pred": pred,
                        "y_score": scores,
                    }
                )
            except Exception as exc:
                item["tuning_error"] = str(exc)

    def _imbalance_diagnostics(self, y_train, problem_type):
        if problem_type != "classification":
            return {"ratio": 1.0, "strategy": "none", "before": {}, "after": {}, "message": "Regression target; class balancing not applicable."}
        counts = pd.Series(y_train).value_counts().sort_index()
        ratio = float(counts.max() / counts.min()) if len(counts) > 1 and counts.min() > 0 else 1.0
        before = {str(k): int(v) for k, v in counts.items()}
        after = {}
        strategy = "none"
        if ratio > 1.5:
            strategy = "class_weight + train-fold resampling"
            after = {str(k): int(counts.max()) for k in counts.index}
        return {
            "ratio": ratio,
            "strategy": strategy,
            "before": before,
            "after": after,
            "raw_y_train": pd.Series(y_train).tolist(),
            "message": f"Imbalance ratio {ratio:.2f}; {'adaptive balancing enabled' if ratio > 1.5 else 'no resampling required'}.",
        }

    def _sampler_for_model(self, name, family, imbalance, problem_type):
        if problem_type != "classification" or imbalance.get("ratio", 1.0) <= 1.5 or not IMBLEARN_AVAILABLE:
            return None
        if family in {"linear", "svm"}:
            return None
        counts = pd.Series(imbalance.get("raw_y_train", [])).value_counts()
        if counts.empty or int(counts.min()) < 2:
            return None
        k_neighbors = max(1, min(3, int(counts.min()) - 1))
        if imbalance.get("ratio", 1.0) >= 2.5 and "Boost" in name:
            return SMOTETomek(smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors), random_state=RANDOM_STATE)
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)

    def _scores(self, model, X_test, problem_type, positive_label):
        if problem_type != "classification":
            return []
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                return proba[:, positive_label if positive_label is not None else 1]
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                scores = np.asarray(scores, dtype=float)
                return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        except Exception:
            return []
        return []

    def _predict_with_threshold(self, model, X_test, y_score, problem_type, positive_label):
        if problem_type == "classification" and positive_label is not None and len(y_score):
            threshold = 0.5
            return np.where(np.asarray(y_score) >= threshold, positive_label, 1 - positive_label)
        return model.predict(X_test)

    def _threshold_optimization(self, y_true, y_score, positive_label):
        y_binary = (np.asarray(y_true) == positive_label).astype(int)
        precision, recall, thresholds = precision_recall_curve(y_binary, y_score)
        rows = []
        best_f1, best_threshold = -1.0, 0.5
        for idx, threshold in enumerate(thresholds):
            p = float(precision[idx])
            r = float(recall[idx])
            f1 = float((2 * p * r) / (p + r + 1e-12))
            rows.append({"threshold": float(threshold), "precision": p, "recall": r, "f1": f1})
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(threshold)
        recall_first = next((row for row in rows if row["recall"] >= 0.7), rows[0] if rows else {})
        return {
            "best_f1_threshold": best_threshold,
            "best_f1": best_f1,
            "recall_70_threshold": recall_first.get("threshold"),
            "points": rows[:: max(len(rows) // 50, 1)] if rows else [],
        }

    def _normalized_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        denom = cm.sum(axis=1, keepdims=True)
        return np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0).tolist()

    def _grid(self, name):
        if "XGB" in name:
            return {"n_estimators": [100, 200, 300], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [3, 5, 7], "subsample": [0.75, 0.9, 1.0]}
        if "LGBM" in name:
            return {"n_estimators": [100, 200, 300], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [20, 31, 50]}
        if "RandomForest" in name or "ExtraTrees" in name:
            return {"n_estimators": [100, 200, 300], "max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]}
        if "CatBoost" in name:
            return {"iterations": [100, 200, 300], "learning_rate": [0.02, 0.05, 0.1], "depth": [4, 6, 8]}
        if "LogisticRegression" in name:
            return {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "saga"]}
        if "GradientBoosting" in name:
            return {"n_estimators": [100, 200, 300], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3, 4]}
        return None

    def _failed(self, name, error, elapsed=0):
        return {
            "name": name,
            "display_name": name.replace("Classifier", "").replace("Regressor", ""),
            "metrics": {},
            "training_time_sec": round(float(elapsed), 3),
            "tuned": False,
            "tuned_params": {},
            "tuned_metrics": {},
            "rank": None,
            "error": error,
        }

    def _scaler_name(self, family):
        return "StandardScaler" if family in {"linear", "svm", "knn"} else "none"

    def _weight_strategy(self, name, imbalance):
        if imbalance.get("ratio", 1.0) <= 1.5:
            return "none"
        if name in {"LogisticRegression", "RandomForestClassifier", "ExtraTreesClassifier", "SVC"}:
            return "class_weight"
        if name == "XGBClassifier":
            return "scale_pos_weight"
        return "none"

    def _notes(self, problem_type, imbalance):
        text = (
            "Categorical features are imputed and one-hot encoded inside train-fold preprocessing. "
            "Numeric features are median-imputed; linear/SVM/KNN models also receive scaling. "
            "Identifier columns are removed before this stage."
        )
        if problem_type == "classification":
            text += f" Class imbalance ratio was {imbalance.get('ratio', 1.0):.2f}; {imbalance.get('strategy', 'none')}."
        return text
