# backend/agents/explainability_agent.py
import joblib
import numpy as np
import traceback
from sklearn.inspection import permutation_importance

from utils.serializer import make_serializable

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ExplainabilityAgent:
    def run(self, model_path: str, X_train, X_test, feature_names: list, model_type: str, problem_type: str) -> dict:
        if not SHAP_AVAILABLE:
            return self._native_importance_fallback(model_path, X_train, X_test, feature_names)

        try:
            model = joblib.load(model_path)
            estimator = model
            names = feature_names
            X_train_sample = X_train.iloc[: min(len(X_train), 200)] if hasattr(X_train, "iloc") else X_train[: min(len(X_train), 200)]
            X_test_sample = X_test.iloc[: min(len(X_test), 100)] if hasattr(X_test, "iloc") else X_test[: min(len(X_test), 100)]
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                preprocessor = model.named_steps["preprocessor"]
                estimator = model.named_steps.get("model", model)
                X_train_sample = preprocessor.transform(X_train_sample)
                X_test_sample = preprocessor.transform(X_test_sample)
                try:
                    names = [str(name) for name in preprocessor.get_feature_names_out()]
                except Exception:
                    names = feature_names
                if "variance" in model.named_steps:
                    selector = model.named_steps["variance"]
                    try:
                        support = selector.get_support()
                        X_train_sample = selector.transform(X_train_sample)
                        X_test_sample = selector.transform(X_test_sample)
                        names = [name for name, keep in zip(names, support) if keep]
                    except Exception:
                        pass

            class_name = estimator.__class__.__name__
            if any(token in class_name for token in ["Forest", "XGB", "LGBM", "CatBoost", "Boosting", "Tree"]):
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X_test_sample)
            elif any(token in class_name for token in ["LogisticRegression", "Ridge", "Lasso", "LinearRegression"]):
                explainer = shap.LinearExplainer(estimator, X_train_sample)
                shap_values = explainer.shap_values(X_test_sample)
            else:
                background = shap.sample(X_train_sample, min(20, len(X_train_sample)))
                explainer = shap.KernelExplainer(estimator.predict, background)
                shap_values = explainer.shap_values(X_test_sample[:20])
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]
            mean_abs = np.abs(shap_values).mean(axis=0)
            importance = sorted(
                [{"feature": str(names[i]), "mean_abs_shap": float(v)} for i, v in enumerate(mean_abs[: len(names)])],
                key=lambda x: x["mean_abs_shap"],
                reverse=True,
            )
            permutation = []
            try:
                pi = permutation_importance(model, X_test.iloc[: min(len(X_test), 200)], model.predict(X_test.iloc[: min(len(X_test), 200)]), n_repeats=5, random_state=42)
                permutation = sorted(
                    [
                        {"feature": str(feature_names[i]), "importance_mean": float(v), "importance_std": float(pi.importances_std[i])}
                        for i, v in enumerate(pi.importances_mean[: len(feature_names)])
                    ],
                    key=lambda x: abs(x["importance_mean"]),
                    reverse=True,
                )
            except Exception:
                permutation = []
            top = importance[:5]
            if len(top) >= 3:
                plain = f"The top 3 most influential features were {top[0]['feature']}, {top[1]['feature']}, and {top[2]['feature']}. {top[0]['feature']} had a mean absolute SHAP value of {top[0]['mean_abs_shap']:.3f}, indicating it has the strongest impact on predictions."
            elif top:
                plain = f"The most influential feature was {top[0]['feature']}."
            else:
                plain = "SHAP ran successfully, but no feature importances were produced."
            insights = self._plain_insights(top)
            return make_serializable(
                {
                    "status": "success",
                    "global_importance": importance,
                    "permutation_importance": permutation,
                    "top_features": [x["feature"] for x in top],
                    "plain_english": plain,
                    "global_insights": insights,
                    "shap_available": True,
                }
            )
        except Exception as exc:
            return self._native_importance_fallback(model_path, X_train, X_test, feature_names, error=str(exc))

    def _native_importance_fallback(self, model_path: str, X_train, X_test, feature_names: list, error: str = "") -> dict:
        """Extract feature importances from model coefficients / tree importances when SHAP is unavailable."""
        try:
            model = joblib.load(model_path)
            estimator = model
            names = list(feature_names)
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                preprocessor = model.named_steps["preprocessor"]
                estimator = model.named_steps.get("model", model)
                try:
                    names = [str(n) for n in preprocessor.get_feature_names_out()]
                except Exception:
                    pass
                if "variance" in model.named_steps:
                    try:
                        support = model.named_steps["variance"].get_support()
                        names = [n for n, keep in zip(names, support) if keep]
                    except Exception:
                        pass
            fallback_importance = []
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
                fallback_importance = sorted(
                    [{"feature": str(names[i]), "mean_abs_shap": float(v)} for i, v in enumerate(importances[: len(names)])],
                    key=lambda x: x["mean_abs_shap"],
                    reverse=True,
                )
            elif hasattr(estimator, "coef_"):
                coefs = np.abs(estimator.coef_[0]) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)
                fallback_importance = sorted(
                    [{"feature": str(names[i]), "mean_abs_shap": float(v)} for i, v in enumerate(coefs[: len(names)])],
                    key=lambda x: x["mean_abs_shap"],
                    reverse=True,
                )
            if fallback_importance:
                top = fallback_importance[:5]
                return make_serializable(
                    {
                        "status": "success",
                        "global_importance": fallback_importance,
                        "permutation_importance": [],
                        "top_features": [x["feature"] for x in top],
                        "plain_english": f"Top features by native importance: {', '.join(x['feature'] for x in top)}.",
                        "global_insights": self._plain_insights(top),
                        "shap_available": False,
                        "warning": "SHAP not installed; using native model importances.",
                    }
                )
        except Exception:
            pass
        reason = f"shap not installed{'; ' + error if error else ''}"
        return {"status": "skipped", "reason": reason, "shap_available": False, "top_features": []}

    def _plain_insights(self, top):
        if not top:
            return []
        insights = []
        for item in top[:5]:
            feature = item["feature"]
            readable = feature.replace("_", " ").replace("categorical__", "").replace("numeric__", "")
            insights.append(
                {
                    "feature": feature,
                    "insight": f"{readable} is a major driver of model decisions and should be reviewed for business validity.",
                    "severity": "HIGH" if item.get("mean_abs_shap", 0) == top[0].get("mean_abs_shap", 0) else "MEDIUM",
                }
            )
        return insights
