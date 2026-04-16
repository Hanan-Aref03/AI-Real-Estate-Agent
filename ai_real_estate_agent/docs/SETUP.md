# Setup Guide

## Prerequisites

- Python 3.9+
- Git
- UV (for dependency management)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai_real_estate_agent
```

### 2. Install Dependencies

Using UV (recommended):

```bash
uv sync
```

Using pip:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the `.env` template and add your API keys:

```bash
# Edit .env with your credentials
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

### 4. Add Machine Learning Models

Place your trained models in the `models/` directory:

- `best_ames_model_random_forest.pkl` - Trained Random Forest model
- `feature_names.pkl` - Feature names from training
- `train_stats.json` - Training statistics

### 5. Start the Application

#### FastAPI Backend

```bash
# Using UV
uv run python -m app.main

# Using Python directly
python -m app.main
```

Server runs on `http://localhost:8000`

#### Streamlit UI (in a separate terminal)

```bash
# Using UV
uv run streamlit run ui/streamlit_app.py

# Using Python directly
streamlit run ui/streamlit_app.py
```

UI runs on `http://localhost:8501`

## Docker Setup

### Build the Image

```bash
docker build -t ai-real-estate-agent .
```

### Run the Container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e ANTHROPIC_API_KEY=your_key \
  -v $(pwd)/models:/app/models \
  ai-real-estate-agent
```

## Troubleshooting

### Module Not Found Error

Ensure you've run `uv sync` or `pip install -r requirements.txt`.

### Model Loading Error

Check that model files exist in the `models/` directory with correct names.

### API Key Error

Verify that API keys are set in `.env` file or environment variables.
