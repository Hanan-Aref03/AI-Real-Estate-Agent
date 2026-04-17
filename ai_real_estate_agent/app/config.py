"""Configuration settings for the application."""

import os

from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini" if GEMINI_API_KEY else "mock")

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
TRAIN_STATS_PATH = os.path.join(MODEL_DIR, "train_stats.json")

# API Settings
API_TITLE = "AI Real Estate Agent"
API_VERSION = "1.0.0"
API_PORT = int(os.getenv("API_PORT", 8000))
