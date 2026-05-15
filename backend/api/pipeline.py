# backend/api/pipeline.py
import json
import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import logging
from agents.evaluator import EvaluatorAgent
from orchestrator.graph import run_pipeline
from utils import progress_store
from utils.serializer import make_serializable
from utils.validation import validate_dataset

logger = logging.getLogger("pipeline")

router = APIRouter()


def _read_dataset(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise HTTPException(status_code=422, detail="File must be CSV or Excel.")


def get_compatible_models(task_type: str, requested_models: list = None):
    compatible = []
    requested_models = requested_models or []
    
    if task_type == "regression":
        allowed = ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor", "ExtraTreesRegressor", "XGBRegressor", "LGBMRegressor", "CatBoostRegressor", "Ridge", "Lasso", "SVR"]
        defaults = ["LinearRegression", "RandomForestRegressor"]
    else:
        allowed = ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier", "ExtraTreesClassifier", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier", "SVC", "KNeighborsClassifier", "DecisionTreeClassifier", "HistGradientBoostingClassifier"]
        defaults = ["LogisticRegression", "RandomForestClassifier"]
        
    for model in requested_models:
        if model in allowed:
            compatible.append(model)
            
    injected = False
    if not compatible:
        compatible = defaults
        injected = True
        
    return compatible, injected


def _background_run(df, goal, target_col, selected_models, user_params, fast_mode, run_id, dataset_path, dataset_name, created_at, run_explainability):
    try:
        run_pipeline(
            df,
            goal,
            target_col,
            selected_models,
            user_params,
            fast_mode,
            run_id,
            progress_callback=lambda agent, status, message: progress_store.update(run_id, agent, status, message),
            dataset_path=str(dataset_path),
            dataset_name=dataset_name,
            created_at=created_at,
            run_explainability=run_explainability,
        )
    except Exception as exc:
        progress_store.update(run_id, "pipeline", "error", str(exc))


@router.post("/run")
async def start_pipeline(
    background_tasks: BackgroundTasks,
    goal: str = Form(...),
    target_col: str = Form(None),
    selected_models: str = Form("[]"),
    fast_mode: bool = Form(False),
    run_explainability: bool = Form(True),
    file: UploadFile = File(...),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=422, detail="File must be CSV or Excel.")
    run_id = str(uuid.uuid4())[:8]
    uploads_dir = Path("./.storage/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "dataset.csv").name
    uploaded_path = uploads_dir / f"{run_id}_{safe_name}"
    uploaded_path.write_bytes(await file.read())
    df = _read_dataset(uploaded_path)
    if df.shape[0] < 50:
        raise HTTPException(status_code=422, detail="Dataset must have at least 50 rows.")
    if df.shape[1] < 2:
        raise HTTPException(status_code=422, detail="Dataset must have at least 2 columns.")
    if target_col and target_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target_col}' not found in dataset. "
                   f"Available columns: {list(df.columns)}"
        )
    # ── Universal Data Validation ───────────────────────
    val_result = validate_dataset(df, target_col)
    if not val_result["valid"]:
        raise HTTPException(
            status_code=400,
            detail=val_result["errors"][0]
        )
    # ── Performance Safeguard ─────────────────────────────
    if len(df) > 100000:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    # ── End pre-flight checks ─────────────────────────────
    try:
        parsed_models = json.loads(selected_models) if selected_models else []
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="selected_models must be a JSON array.")
        
    # ── Task-Type-Aware Model Filtering Failsafe ──────────
    task_type = val_result["task_type"]
    filtered_models, injected = get_compatible_models(task_type, parsed_models)
    
    logger.info(f"Task type detected: {task_type}")
    logger.info(f"Requested models: {parsed_models}")
    logger.info(f"Filtered models: {filtered_models}")
    if injected:
        logger.warning(f"Incompatible or empty model list received. Injected defaults: {filtered_models}")
    
    created_at = time.time()
    
    # Pass the filtered models to the graph state
    progress_store.update(run_id, "planner", "running", "Starting pipeline...")
    background_tasks.add_task(_background_run, df, goal, target_col or None, filtered_models, {}, fast_mode, run_id, uploaded_path, safe_name, created_at, run_explainability)
    return {"run_id": run_id, "status": "started", "task_type": task_type, "requested_models": parsed_models, "filtered_models": filtered_models}


@router.get("/status/{run_id}")
def pipeline_status(run_id: str):
    status = progress_store.get(run_id)
    run_path = Path(f"./.storage/runs/{run_id}.json")
    if status.get("status") == "not_found" and run_path.exists():
        data = json.loads(run_path.read_text(encoding="utf-8"))
        stored_status = data.get("status", "completed")
        response = {
            "status": stored_status, 
            "progress_pct": 100 if stored_status == "completed" else 0, 
            "current_agent": "pipeline", 
            "completed_agents": [], 
            "logs": [], 
            "error": data.get("error")
        }
        if stored_status == "failed":
            response["traceback"] = data.get("traceback")
            err_str = str(data.get("error", "")).lower()
            if "target column" in err_str and "less than 2" in err_str:
                response["remediation_suggestion"] = "Please select a different target column with at least 2 unique values (e.g., categorical or numeric)."
            elif "not found" in err_str or "keyerror" in err_str:
                response["remediation_suggestion"] = "Ensure the target column is exactly as it appears in the CSV header."
            elif "empty" in err_str:
                response["remediation_suggestion"] = "Dataset is empty after cleaning. Check if too many missing values caused all rows/columns to be dropped."
            elif "no models trained successfully" in err_str:
                response["remediation_suggestion"] = "All machine learning models failed to train. Check debug logs or results to see per-model errors."
                response["ml_results"] = data.get("ml_results", {})
            else:
                response["remediation_suggestion"] = "Review the traceback in the logs or verify the dataset structure and target column."
        return response
    return status


@router.get("/result/{run_id}")
def pipeline_result(run_id: str):
    path = Path(f"./.storage/runs/{run_id}.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    ml = data.get("ml_results", {})
    evaluation = data.get("evaluation_results", {})
    if ml and evaluation.get("status") == "error":
        problem_type = ml.get("problem_type") or data.get("problem_type", "classification")
        data["evaluation_results"] = EvaluatorAgent().run(ml, ml.get("y_test", []), ml.get("y_pred", []), problem_type)
    return JSONResponse(content=make_serializable(data))


@router.get("/debug/{run_id}")
def pipeline_debug(run_id: str):
    path = Path(f"./.storage/runs/{run_id}.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return JSONResponse(content=data)
