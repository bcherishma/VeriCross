from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import evaluate
from app.utils.logger import setup_logging

setup_logging()

app = FastAPI(title="VeriCross", description="Hallucination Detection Suite")

# Add CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate.router)