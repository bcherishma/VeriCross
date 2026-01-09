from app.routers import evaluate
from app.utils.logger import setup_logging

setup_logging()

app = evaluate.app  # Assuming we move the app creation