# InsightForge Setup

InsightForge is an AI-powered autonomous data science platform with a FastAPI backend and a React/Vite frontend.

## Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Optional Features

```bash
pip install lightgbm catboost shap faiss-cpu sentence-transformers imbalanced-learn kaleido
```

## Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq key:

```bash
GROQ_API_KEY=gsk-your-groq-key-here
LLM_MODEL_NAME=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0
FAST_MODE=false
STORAGE_PATH=./storage
```

## Run Backend

```bash
uvicorn main:app --reload --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: http://localhost:5173
