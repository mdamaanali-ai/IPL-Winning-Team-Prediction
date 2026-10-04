from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MATCHES_FILE = DATA_DIR / "matches.csv"
MODEL_FILE = MODEL_DIR / "ipl_winner_model.joblib"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
CONFUSION_FILE = OUTPUT_DIR / "confusion_matrix.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/avinashyadav16/ipl-analytics/"
    "main/matches_2008-2024.csv"
)
