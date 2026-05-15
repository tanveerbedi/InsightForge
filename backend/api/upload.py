import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

UPLOAD_DIR = Path(".storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    safe_name = Path(filename).name
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = UPLOAD_DIR / unique_name

    try:
        contents = await file.read()
        file_path.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}")

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}")

    if df.empty:
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    df.columns = df.columns.astype(str).str.strip()

    if len(df.columns) < 2:
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="CSV needs at least 2 columns (features + target).")

    preview_df = df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), other=None)
    preview_rows = preview_df.to_dict(orient="records")
    columns = df.columns.tolist()

    return {
        "filename": safe_name,
        "file_path": str(file_path),
        "columns": columns,
        "preview_rows": preview_rows,
        "n_rows": len(df),
        "n_cols": len(columns),
    }
