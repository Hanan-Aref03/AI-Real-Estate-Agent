# Project Structure

```
ai_real_estate_agent/
├── app/                           # Main FastAPI application
│   ├── __init__.py
│   ├── main.py                    # FastAPI app initialization
│   ├── config.py                  # Configuration settings
│   ├── schemas.py                 # Pydantic models
│   ├── model_loader.py            # ML model management
│   ├── llm_client.py              # LLM API client
│   └── routers/                   # API route modules
│       ├── __init__.py
│       ├── predictions.py         # Price prediction endpoints
│       └── queries.py             # LLM query endpoints
│
├── ui/                            # User interfaces
│   └── streamlit_app.py           # Streamlit web interface
│
├── models/                        # Machine learning models directory
│   ├── best_ames_model_random_forest.pkl   # Trained model
│   ├── feature_names.pkl          # Feature names
│   └── train_stats.json           # Training statistics
│
├── docs/                          # Documentation
│   ├── SETUP.md                   # Setup and installation guide
│   ├── API_DOCS.md                # API documentation
│   ├── PROJECT_STRUCTURE.md       # This file
│   └── DEPLOYMENT.md              # Deployment guide
│
├── pyproject.toml                 # UV/setuptools configuration
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── .dockerignore                  # Docker ignore rules
├── .gitignore                     # Git ignore rules
├── .env                           # Environment variables (not in repo)
└── README.md                      # Main project documentation
```

## Directory Descriptions

### `/app` - Application Core

Contains the FastAPI backend application with route handlers, data models, ML model integration, and LLM clients.

- **main.py** - FastAPI app setup, middleware, and initialization
- **config.py** - Environment configuration and constants
- **schemas.py** - Pydantic models for request/response validation
- **model_loader.py** - ML model lifecycle management
- **llm_client.py** - LLM provider integration (OpenAI/Anthropic)
- **routers/** - Modular API endpoint definitions

### `/ui` - User Interfaces

Frontend applications for user interaction.

- **streamlit_app.py** - Interactive web UI for price prediction and Q&A

### `/models` - ML Artifacts

Stores trained models and training metadata.

- **.pkl files** - Serialized scikit-learn models
- **train_stats.json** - Training metrics and feature information

### `/docs` - Documentation

Comprehensive guides for setup, usage, and deployment.

## Module Organization

### Routers

Routes are organized by feature:

- **predictions.py** - Price prediction endpoints
- **queries.py** - LLM query endpoints

This modular approach allows for easy scaling and adding new features.

### Models

- **Pydantic Models** - Request/response validation
- **ML Models** - Prediction models stored as pickles
- **Config Models** - Application settings

## Data Flow

```
User Request
    ↓
Streamlit UI or HTTP Client
    ↓
FastAPI Router (app/routers/)
    ↓
Business Logic (models, clients)
    ↓
ML Model or LLM API
    ↓
Response
```
