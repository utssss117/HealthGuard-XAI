import os
import json
import logging
from datetime import datetime

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("healthguard")

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "logs")
LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")

def log_prediction_to_jsonl(biomarkers: dict, risk_probability: float, risk_level: str):
    """Log prediction inputs and outputs structured as JSON Lines for future drift monitoring."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_features": biomarkers,
        "risk_probability": risk_probability,
        "risk_level": risk_level
    }
    
    # Standard application logging
    logger.info(f"Prediction Record: {json.dumps(log_entry)}")
    
    # Append to structured predictions.jsonl file
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write prediction log: {e}")
