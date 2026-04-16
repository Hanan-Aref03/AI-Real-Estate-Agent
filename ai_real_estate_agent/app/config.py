"""Configuration settings for the application."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_ames_model_random_forest.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
TRAIN_STATS_PATH = os.path.join(MODEL_DIR, "train_stats.json")

# API Settings
API_TITLE = "AI Real Estate Agent"
API_VERSION = "1.0.0"
API_PORT = int(os.getenv("API_PORT", 8000))
