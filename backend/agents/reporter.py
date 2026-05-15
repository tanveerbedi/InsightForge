# backend/agents/reporter.py
import traceback

class ReporterAgent:
    def run(self, plan: dict, cleaning: dict, eda: dict, ml: dict, explainability: dict, evaluation: dict, dataset_name: str) -> dict:
        try:
            best_model = ml.get("best_model_name", "the selected model")
            problem_type = ml.get("problem_type", "machine learning")
            metrics = ml.get("best_metrics", {})
            primary_value = metrics.get("churn_recall") if problem_type == "classification" else metrics.get("r2")
            quality = cleaning.get("data_quality_score", 0)
            shape = eda.get("shape", ["unknown", "unknown"])
            churn_text = f"Minority-class recall was {primary_value:.4f}; weighted F1 was {metrics.get('f1_weighted', 0):.4f}. " if problem_type == "classification" and isinstance(primary_value, (int, float)) else ""
            score_text = churn_text or (f"The best score was {primary_value:.4f}. " if isinstance(primary_value, (int, float)) else "")
            executive_summary = f"InsightForge analyzed {dataset_name} with {shape[0]} rows and {shape[1]} columns. The workflow identified a {problem_type} task and selected {best_model} as the strongest model. {score_text}Data quality scored {quality}/100 after automated cleaning."
            key_findings = [
                f"Data cleaning removed {cleaning.get('rows_removed', 0)} rows and {cleaning.get('cols_removed', 0)} columns.",
                f"Missing value handling fixed {cleaning.get('missing_fixed', 0)} cells.",
                f"The best model was {best_model}.",
                ml.get("why_best", "Model comparison completed successfully."),
                explainability.get("plain_english", "Explainability was unavailable for this run."),
            ]
            if cleaning.get("identifier_columns"):
                key_findings.insert(0, f"Identifier leakage prevention removed: {', '.join([item['column'] for item in cleaning['identifier_columns']])}.")
            if eda.get("insights"):
                key_findings.append(eda["insights"][0].get("description", "EDA completed."))
            recommendations = [
                "Prioritize minority-class recall, PR-AUC, and threshold tradeoffs over weighted F1 for business decisions.",
                "Validate the selected model on fresh, unseen data before production deployment.",
                "Monitor data drift and retrain when feature distributions or target balance change materially.",
            ]
            if ml.get("class_imbalance_handled"):
                recommendations[0] = "Track class-level performance because imbalance was detected and adaptive balancing was used during training."
            minority_recall = metrics.get("churn_recall")
            if isinstance(minority_recall, (int, float)) and minority_recall < 0.65:
                recommendations.insert(0, "Tune the classification threshold with business cost assumptions before acting on predictions.")
            verdict = "Strong data quality after cleaning." if quality >= 80 else "Moderate data quality; review cleaning decisions before relying on downstream metrics." if quality >= 60 else "Low data quality; collect or repair source data before using model outputs for important decisions."
            next_steps = ["Export the report for stakeholder review.", "Review threshold optimization in Evaluation.", "Schedule a holdout validation pass with new data."]
            top_features = explainability.get("top_features") or []
            if top_features:
                next_steps.append(f"Investigate business meaning for top feature '{top_features[0]}'.")
            return {
                "status": "success",
                "executive_summary": executive_summary,
                "business_insights": key_findings[:6],
                "key_findings": key_findings[:6],
                "recommendations": recommendations[:4],
                "risk_analysis": [
                    "Weighted metrics can hide minority-class prediction errors.",
                    "Model quality must be revalidated after data drift or campaign policy changes.",
                ],
                "deployment_readiness": "Ready for offline stakeholder review; not production-ready until threshold policy, monitoring, and fresh holdout validation are completed.",
                "model_limitations": [
                    "Historical churn labels may encode past retention policies.",
                    "Correlation and SHAP explanations are directional diagnostics, not causal proof.",
                ],
                "model_conclusion": evaluation.get("recommendation") or f"{best_model} is the recommended model for this dataset.",
                "data_quality_verdict": verdict,
                "next_steps": next_steps,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
