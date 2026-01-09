# VeriCross

An automated evaluation suite to detect hallucinations in generative model summaries across languages.

## Architecture

- **backend/**: FastAPI application with ML models
- **frontend/**: Simple web interface for evaluation

## Features

- Semantic Entailment Scoring using RoBERTa-large-MNLI
- Entity Alignment for Named Entities using spaCy
- Language-Agnostic Architecture with multilingual embeddings (LaBSE)
- Visual Confidence Heatmap based on sentence similarity
- Real-time evaluation via WebSockets and REST API

## Tech Stack

- **Backend**: FastAPI, WebSockets, Hugging Face Transformers, PyTorch, Sentence Transformers, BERTScore, spaCy
- **Frontend**: HTML, JavaScript, Nginx

## Installation & Running

### Using Docker Compose (Recommended)
```bash
docker-compose up --build
```
- Frontend: http://localhost
- Backend API: http://localhost:8000

### Manual Setup

#### Backend
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm xx_ent_wiki_sm
uvicorn app.main:app --reload
```

#### Frontend
```bash
cd frontend
python -m http.server 8080  # or use any static server
```

## Usage

- **Web Interface**: Open http://localhost and fill the form
- **REST API**: POST `http://localhost:8000/evaluate`
- **WebSocket**: Connect to `ws://localhost:8000/ws/evaluate`

## Production Deployment

- Use Docker Compose for containerized deployment
- Models are cached for performance
- Logging with loguru for monitoring
- Environment variables for configuration
