import os
import sys
import shutil
import requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.config import load_config
from scripts.utils import setup_logging

logger = setup_logging("download_datasets")

def download_file(url: str, dest_path: str):
    """Downloads a file from a URL with a progress bar."""
    logger.info(f"Downloading from {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path))
    with open(dest_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    logger.info("Download completed.")

def prepare_davidson(raw_dir: str):
    """Prepares Davidson dataset. Tries local workspace file first, then url."""
    dest_path = os.path.join(raw_dir, "davidson", "davidson.csv")
    if os.path.exists(dest_path):
        logger.info("Davidson dataset already exists in raw folder.")
        return
        
    # Check if a copy exists in legacy root dataset folder
    legacy_path = os.path.join(project_root, "dataset", "davidson.csv")
    if os.path.exists(legacy_path):
        logger.info(f"Copying Davidson dataset from legacy folder: {legacy_path}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(legacy_path, dest_path)
        return

    url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
    try:
        download_file(url, dest_path)
    except Exception as e:
        logger.error(f"Failed to download Davidson dataset: {e}")
        logger.info("Please manually place the labeled_data.csv into dataset/raw/davidson/davidson.csv")

def prepare_hatexplain(raw_dir: str):
    """Prepares HateXplain dataset. Tries local workspace file first, then url."""
    dest_path = os.path.join(raw_dir, "hatexplain", "hatexplain.json")
    if os.path.exists(dest_path):
        logger.info("HateXplain dataset already exists in raw folder.")
        return

    legacy_path = os.path.join(project_root, "dataset", "hatexplain.json")
    if os.path.exists(legacy_path):
        logger.info(f"Copying HateXplain dataset from legacy folder: {legacy_path}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(legacy_path, dest_path)
        return

    url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        logger.info(f"Downloading HateXplain from {url}...")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        logger.info("HateXplain downloaded.")
    except Exception as e:
        logger.error(f"Failed to download HateXplain JSON: {e}")
        logger.info("Please manually place HateXplain dataset.json into dataset/raw/hatexplain/hatexplain.json")

def prepare_olid(raw_dir: str):
    """Prepares OLID dataset from HuggingFace."""
    dest_path = os.path.join(raw_dir, "olid", "olid.csv")
    if os.path.exists(dest_path):
        logger.info("OLID dataset already exists in raw folder.")
        return

    logger.info("Loading OLID dataset from HuggingFace cardiffnlp/tweet_eval...")
    try:
        dataset = load_dataset("cardiffnlp/tweet_eval", "offensive")
        records = []
        for split in ["train", "validation", "test"]:
            for item in dataset[split]:
                records.append({
                    "text": item["text"],
                    "label": item["label"]  # 0: non-offensive, 1: offensive
                })
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        df.to_csv(dest_path, index=False, encoding="utf-8")
        logger.info(f"OLID saved to {dest_path}. Size: {len(df)} rows.")
    except Exception as e:
        logger.error(f"Failed to load OLID from HF: {e}")
        logger.info("Please manually download the OLID dataset and place the CSV in dataset/raw/olid/olid.csv")

def prepare_jigsaw(raw_dir: str):
    """Prepares Jigsaw dataset from HuggingFace/Mirror or fallback instructions."""
    dest_path = os.path.join(raw_dir, "jigsaw", "train.csv")
    if os.path.exists(dest_path):
        logger.info("Jigsaw dataset already exists in raw folder.")
        return

    # Try HuggingFace mirror first
    logger.info("Attempting to load Jigsaw dataset from HuggingFace (thesofakillers/jigsaw-toxic-comment-classification-challenge)...")
    try:
        dataset = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge", trust_remote_code=True)
        df = pd.DataFrame(dataset["train"])
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        df.to_csv(dest_path, index=False, encoding="utf-8")
        logger.info(f"Jigsaw dataset saved to {dest_path}. Size: {len(df)} rows.")
        return
    except Exception as e:
        logger.warning(f"Failed to download Jigsaw from HF: {e}. Trying public mirror url...")

    # Try github mirror url fallback
    mirror_url = "https://raw.githubusercontent.com/trip3ee/Jigsaw-Toxic-Comment-Classification-Challenge/master/data/train.csv"
    try:
        download_file(mirror_url, dest_path)
        logger.info("Jigsaw dataset successfully downloaded from mirror.")
        return
    except Exception as e:
        logger.error(f"Mirror download failed: {e}")

    logger.error("Jigsaw dataset download failed.")
    logger.critical(
        "KAGGLE DOWNLOAD REQUIRED FOR JIGSAW:\n"
        "1. Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data\n"
        "2. Download 'train.csv.zip'\n"
        "3. Unzip and place the 'train.csv' file inside 'dataset/raw/jigsaw/train.csv'\n"
    )

def prepare_civil_comments(raw_dir: str):
    """Prepares Civil Comments dataset using streaming to avoid massive download size."""
    dest_path = os.path.join(raw_dir, "civil_comments", "train.csv")
    if os.path.exists(dest_path):
        logger.info("Civil Comments dataset already exists in raw folder.")
        return

    logger.info("Attempting to stream Civil Comments dataset from HuggingFace google/civil_comments...")
    try:
        dataset = load_dataset("google/civil_comments", split="train", streaming=True)
        records = []
        max_samples = 50000
        logger.info(f"Streaming first {max_samples} comments from google/civil_comments...")
        
        for idx, item in enumerate(tqdm(dataset, total=max_samples, desc="Streaming Civil Comments")):
            if idx >= max_samples:
                break
            records.append({
                "text": item["text"],
                "toxicity": item["toxicity"],
                "severe_toxicity": item["severe_toxicity"],
                "obscene": item["obscene"],
                "threat": item["threat"],
                "insult": item["insult"],
                "identity_attack": item["identity_attack"],
                "sexual_explicit": item.get("sexual_explicit", 0.0)
            })
            
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        df.to_csv(dest_path, index=False, encoding="utf-8")
        logger.info(f"Civil Comments saved to {dest_path}. Size: {len(df)} rows.")
        return
    except Exception as e:
        logger.error(f"Failed to stream Civil Comments from HF: {e}")

    logger.critical(
        "KAGGLE DOWNLOAD REQUIRED FOR CIVIL COMMENTS:\n"
        "1. Go to: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data\n"
        "2. Download 'train.csv.zip'\n"
        "3. Unzip and place the 'train.csv' file inside 'dataset/raw/civil_comments/train.csv'\n"
    )

def main():
    config = load_config()
    raw_dir = config.dataset.raw_dir
    os.makedirs(raw_dir, exist_ok=True)
    
    logger.info("--- STARTING DATASET DOWNLOAD PIPELINE ---")
    prepare_davidson(raw_dir)
    prepare_hatexplain(raw_dir)
    prepare_olid(raw_dir)
    prepare_jigsaw(raw_dir)
    prepare_civil_comments(raw_dir)
    logger.info("--- DATASET DOWNLOAD PIPELINE COMPLETED ---")

if __name__ == "__main__":
    main()
