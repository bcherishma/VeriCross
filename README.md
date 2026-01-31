# VeriCross

An automated evaluation suite to detect hallucinations in generative model summaries across languages. VeriCross uses advanced ML models to evaluate the semantic similarity, entity alignment, and overall confidence between source texts and their summaries, supporting 20+ languages including Indian languages.

## 🌟 Features

- **Automatic Language Detection**: Detects source and summary languages automatically
- **Cross-lingual Semantic Matching**: Uses LaBSE embeddings for multilingual semantic similarity
- **Semantic Entailment Scoring**: Uses RoBERTa-large-MNLI for entailment detection
- **Entity Alignment**: Matches named entities across languages (e.g., "Deutschland" ↔ "Germany")
- **Visual Confidence Heatmap**: Interactive Chart.js bar chart showing sentence-level confidence scores
- **Real-time Evaluation**: REST API and WebSocket support
- **Indian Language Support**: Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Bengali, Urdu, and more

## 🏗️ Architecture

- **Backend**: FastAPI application with ML models
- **Frontend**: Modern web interface with Chart.js visualizations
- **Models**: 
  - LaBSE (Language-agnostic BERT Sentence Embeddings) for cross-lingual similarity
  - RoBERTa-large-MNLI for semantic entailment
  - spaCy multilingual NER for entity extraction

## 🛠️ Tech Stack

- **Backend**: FastAPI, WebSockets, Hugging Face Transformers, PyTorch, Sentence Transformers, BERTScore, spaCy, langdetect
- **Frontend**: HTML, JavaScript, Chart.js, Modern CSS
- **ML Models**: RoBERTa-large-MNLI, LaBSE, spaCy multilingual models

## 📦 Installation & Running

### Prerequisites

- Python 3.10, 3.11, or 3.12 (Python 3.13 not supported due to dependency compatibility)
- pip

### Quick Start (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd VeriCross/VeriCross
   ```

2. **Run the backend**:
   ```bash
   ./run_backend.sh
   ```
   This will:
   - Create a virtual environment
   - Install all dependencies
   - Download required spaCy models
   - Start the FastAPI server on http://localhost:8000

3. **Run the frontend** (in a new terminal):
   ```bash
   ./run_frontend.sh
   ```
   This starts the frontend server on http://localhost:8080

4. **Open your browser**: Navigate to http://localhost:8080

### Manual Setup

#### Backend

```bash
cd VeriCross/VERICROSS
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm xx_ent_wiki_sm
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend

```bash
cd VeriCross/VeriCross/frontend
python3 -m http.server 8080
```

### Using Docker Compose

```bash
docker-compose up --build
```

## 🚀 Usage

### Web Interface

1. Open http://localhost:8080 in your browser
2. Enter your source text (any language)
3. Enter your summary text (any language)
4. Click "Evaluate Summary"
5. Languages are automatically detected
6. View results with:
   - Semantic entailment score
   - Entity matches and mismatches
   - Interactive confidence heatmap chart

### WebSocket

Connect to `ws://localhost:8000/ws/evaluate` for real-time evaluation.

## 🌍 Supported Languages

### European Languages
- English (en), German (de), French (fr), Spanish (es), Italian (it), Portuguese (pt), Russian (ru)

### Asian Languages
- Chinese (zh), Japanese (ja), Korean (ko), Arabic (ar)

### Indian Languages
- Hindi (hi), Tamil (ta), Telugu (te), Kannada (kn), Malayalam (ml)
- Marathi (mr), Gujarati (gu), Punjabi (pa), Bengali (bn), Urdu (ur)
- Odia (or), Assamese (as), Nepali (ne), Sinhala (si)

## 📊 How It Works

1. **Language Detection**: Automatically detects the language of source and summary texts
2. **Semantic Entailment**: Uses RoBERTa-large-MNLI to determine if the summary entails, contradicts, or is neutral to the source
3. **Entity Extraction**: Extracts named entities (persons, locations, organizations) from both texts
4. **Cross-lingual Entity Matching**: Uses LaBSE embeddings to match entities across languages
5. **Semantic Similarity**: Calculates overall semantic similarity using multilingual embeddings
6. **Confidence Scoring**: Combines entailment, entity alignment, and semantic similarity for overall confidence
7. **Visualization**: Displays results with interactive charts and color-coded entity matches

## 🔧 Configuration

### Environment Variables

- `ENV`: Set to `development` or `production` (default: `development`)

### Model Caching

Models are automatically downloaded and cached on first use:
- RoBERTa-large-MNLI: ~1.4GB
- LaBSE: ~420MB
- spaCy models: ~50-100MB each

## 📝 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing

```bash
cd VeriCross/VeriCross
source venv/bin/activate
pytest tests/
```

## 🐛 Troubleshooting

### Python Version Issues

If you encounter build errors, ensure you're using Python 3.10, 3.11, or 3.12:
```bash
python3.10 --version  # Should show 3.10.x
```

### spaCy Model Download Issues

If spaCy models fail to download:
```bash
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
```

Or install via pip:
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
pip install https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.7.0/xx_ent_wiki_sm-3.7.0-py3-none-any.whl
```

### Port Already in Use

If port 8000 or 8080 is already in use:
- Backend: Change port in `run_backend.sh` or use `uvicorn app.main:app --port 8001`
- Frontend: Use `python3 -m http.server 8081`


## 📧 Contact

cherishmawork@gmail.com
https://www.linkedin.com/in/cherishma-bodapati-940158258

