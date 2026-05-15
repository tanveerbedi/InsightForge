import re

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SuggestRequest(BaseModel):
    file_path: str


def score_column(col: pd.Series, name: str, n_rows: int) -> dict:
    score = 0
    n_unique = int(col.nunique())
    null_pct = float(col.isnull().mean())
    unique_ratio = n_unique / max(n_rows, 1)
    name_lower = name.lower().replace("_", "").replace(" ", "")

    if n_unique <= 1:
        return {
            "column": name,
            "col": name,
            "score": -999,
            "reason": "Constant - only 1 unique value",
            "task_type": "none",
            "task": "none",
            "n_unique": n_unique,
            "null_pct": round(null_pct, 4),
        }
    if unique_ratio > 0.9 and n_unique > 20:
        return {
            "column": name,
            "col": name,
            "score": -999,
            "reason": "ID column - every value is unique",
            "task_type": "none",
            "task": "none",
            "n_unique": n_unique,
            "null_pct": round(null_pct, 4),
        }
    if null_pct > 0.5:
        return {
            "column": name,
            "col": name,
            "score": -999,
            "reason": f"Too many nulls ({null_pct:.0%})",
            "task_type": "none",
            "task": "none",
            "n_unique": n_unique,
            "null_pct": round(null_pct, 4),
        }

    if col.dtype == object:
        sample = str(col.dropna().iloc[0]) if len(col.dropna()) > 0 else ""
        if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}", sample):
            return {
                "column": name,
                "col": name,
                "score": -999,
                "reason": "Date column - not a target",
                "task_type": "none",
                "task": "none",
                "n_unique": n_unique,
                "null_pct": round(null_pct, 4),
            }

    bad_names = [
        "id", "uid", "uuid", "name", "key", "code", "ref", "date", "time",
        "timestamp", "created", "updated", "index", "row", "no", "number",
        "phone", "email", "address", "url", "link", "image", "photo", "file",
    ]
    for bad in bad_names:
        if name_lower == bad or name_lower.endswith(bad) or name_lower.startswith(bad):
            score -= 40
            break

    good_names = [
        "target", "label", "class", "churn", "survived", "outcome",
        "result", "status", "fraud", "default", "diagnosis", "disease",
        "price", "cost", "salary", "score", "rating", "value", "amount",
        "fare", "charge", "quality", "grade", "category", "type",
    ]
    for good in good_names:
        if good in name_lower:
            score += 35
            break

    if n_unique == 2:
        score += 50
        task = "classification"
        reason = "Binary column (2 classes) - ideal classification target"
    elif 3 <= n_unique <= 15:
        score += 38
        task = "classification"
        reason = f"Low cardinality ({n_unique} classes) - good classification target"
    elif str(col.dtype) in ["float64", "float32"]:
        score += 33
        task = "regression"
        reason = f"Continuous float ({n_unique} unique) - good regression target"
    elif str(col.dtype) in ["int64", "int32"] and n_unique > 20:
        score += 28
        task = "regression"
        reason = f"Integer with many values ({n_unique}) - regression target"
    else:
        score += 10
        task = "classification"
        reason = f"Moderate target candidate ({n_unique} unique values)"

    score -= null_pct * 15

    return {
        "column": name,
        "col": name,
        "score": round(float(score), 2),
        "reason": reason,
        "task_type": task,
        "task": task,
        "n_unique": n_unique,
        "null_pct": round(null_pct, 4),
    }


@router.post("/suggest")
async def suggest(request: SuggestRequest):
    try:
        df = pd.read_csv(request.file_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read file: {exc}")

    n_rows = len(df)
    scores = [score_column(df[c], c, n_rows) for c in df.columns]
    valid = [s for s in scores if s["score"] > -999]
    valid.sort(key=lambda x: x["score"], reverse=True)

    if not valid:
        raise HTTPException(
            status_code=400,
            detail="No suitable target column found. Dataset may only contain ID columns, dates, or constants.",
        )

    best = valid[0]
    target = best["column"]
    task = best["task_type"]
    features = [c for c in df.columns if c != target]
    n_feat = len(features)

    num_feats = df[features].select_dtypes(include="number").columns.tolist()[:2]
    cat_feats = df[features].select_dtypes(exclude="number").columns.tolist()[:2]
    sample_feats = (num_feats + cat_feats)[:3]
    feat_str = ", ".join(sample_feats) if sample_feats else "the available features"
    if n_feat > 3:
        feat_str += f" and {n_feat - 3} more features"

    target_display = target.replace("_", " ").replace("-", " ").title()
    n_classes = df[target].nunique()

    if task == "classification":
        goal = (
            f"Predict {'whether' if n_classes == 2 else 'which category of'} "
            f"{target_display} using {feat_str}. "
            f"Find the most important features and identify the best "
            f"classification model for this dataset."
        )
    else:
        goal = (
            f"Predict the value of {target_display} based on "
            f"{feat_str}. Analyze which features most influence "
            f"{target_display} and build an accurate regression model."
        )

    return {
        "recommended_target": target,
        "task_type": task,
        "suggested_goal": goal,
        "confidence": "high" if best["score"] >= 60 else "medium" if best["score"] >= 30 else "low",
        "reason": best["reason"],
        "all_scores": scores,
    }
