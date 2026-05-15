# backend/main.py
import traceback
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from api.chat import router as chat_router
from api.export import router as export_router
from api.health import router as health_router
from api.history import router as history_router
from api.pipeline import router as pipeline_router
from api.suggest import router as suggest_router
from api.upload import router as upload_router

load_dotenv()

app = FastAPI(title="InsightForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_URL", "http://localhost:5173"),
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc()[-1000:],
        },
    )


app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(suggest_router, prefix="/api", tags=["suggest"])
app.include_router(pipeline_router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(export_router, prefix="/api/export", tags=["export"])
app.include_router(history_router, prefix="/api/history", tags=["history"])
app.include_router(health_router, prefix="/api", tags=["health"])


@app.get("/")
def root():
    return {"message": "InsightForge API running"}
