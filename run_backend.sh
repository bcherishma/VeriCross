#!/bin/bash
# Run the FastAPI backend server

cd "$(dirname "$0")"

# Use Python 3.10 if available, otherwise fall back to python3
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
else
    PYTHON_CMD=python3
    echo "Warning: Using default python3. If you encounter build errors, install Python 3.10, 3.11, or 3.12"
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Download spacy models separately with error handling
    echo "Downloading spacy models..."
    # Try direct pip install first (more reliable)
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl 2>/dev/null || \
        python -m spacy download en_core_web_sm || \
        echo "Warning: Failed to download en_core_web_sm. Run manually: python -m spacy download en_core_web_sm"
    
    pip install https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.7.0/xx_ent_wiki_sm-3.7.0-py3-none-any.whl 2>/dev/null || \
        python -m spacy download xx_ent_wiki_sm || \
        echo "Warning: Failed to download xx_ent_wiki_sm. Run manually: python -m spacy download xx_ent_wiki_sm"
    
    touch venv/.installed
fi

# Run the backend
echo "Starting FastAPI backend on http://localhost:8000"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
