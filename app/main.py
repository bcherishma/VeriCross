from fastapi import FastAPI
from app.routers import evaluate
from app.utils.logger import setup_logging

setup_logging()

app = FastAPI(title="VeriCross", description="Hallucination Detection Suite")

app.include_router(evaluate.router)