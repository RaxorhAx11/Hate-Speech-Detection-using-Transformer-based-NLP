import os
import sys
import json
import logging
from typing import Dict, Any

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

def setup_logging(name: str = "dataset_pipeline") -> logging.Logger:
    """Sets up unified logging for the dataset pipeline."""
    # Ensure logs folder exists
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(
            os.path.join(logs_dir, "pipeline.log"), encoding="utf-8"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def save_json(data: Any, filepath: str) -> None:
    """Saves data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_json(filepath: str) -> Any:
    """Loads data from a JSON file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_markdown(content: str, filepath: str) -> None:
    """Saves content to a markdown file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
