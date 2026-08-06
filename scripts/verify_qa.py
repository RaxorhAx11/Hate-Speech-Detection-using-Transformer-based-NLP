import os
import sys
import pandas as pd
import json

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.config import load_config
from scripts.utils import setup_logging

logger = setup_logging("verify_qa")

def check_utf8(filepath: str) -> bool:
    """Verifies that the file is valid UTF-8."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False

def verify_file(filepath: str, name: str) -> bool:
    """Runs data quality checks on a CSV split."""
    if not os.path.exists(filepath):
        logger.error(f"{name} split not found at {filepath}")
        return False
        
    # Check UTF-8
    if not check_utf8(filepath):
        logger.error(f"{name} split has invalid UTF-8 encoding.")
        return False
        
    df = pd.read_csv(filepath)
    
    # Check null text or empty text
    null_texts = df["text"].isnull().sum()
    empty_texts = (df["text"].astype(str).str.strip() == "").sum()
    
    if null_texts > 0:
        logger.error(f"{name} has {null_texts} null text records.")
        return False
        
    if empty_texts > 0:
        logger.error(f"{name} has {empty_texts} empty/whitespace text records.")
        return False
        
    # Check valid labels {0, 1, 2}
    invalid_labels = df[~df["label"].isin([0, 1, 2])]
    if len(invalid_labels) > 0:
        logger.error(f"{name} has {len(invalid_labels)} invalid labels: {invalid_labels['label'].unique()}")
        return False
        
    logger.info(f"{name} validation PASSED. Row count: {len(df)}. Labels: {df['label'].value_counts().to_dict()}")
    return True

def main():
    config = load_config()
    data_dir = config.dataset.data_dir
    merged_dir = config.dataset.merged_dir
    
    logger.info("--- STARTING QUALITY ASSURANCE CHECKS ---")
    
    success = True
    
    # Verify outputs
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    # 1. Check if files exist and validation runs
    success &= verify_file(train_path, "Train")
    success &= verify_file(val_path, "Validation")
    success &= verify_file(test_path, "Test")
    
    # 2. Check for leakage (intersection of texts)
    if success:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        train_texts = set(train_df["text"].astype(str).tolist())
        val_texts = set(val_df["text"].astype(str).tolist())
        test_texts = set(test_df["text"].astype(str).tolist())
        
        leak_train_val = train_texts.intersection(val_texts)
        leak_train_test = train_texts.intersection(test_texts)
        leak_val_test = val_texts.intersection(test_texts)
        
        if len(leak_train_val) > 0 or len(leak_train_test) > 0 or len(leak_val_test) > 0:
            logger.error(f"DATA LEAKAGE FOUND! Train-Val: {len(leak_train_val)}, Train-Test: {len(leak_train_test)}, Val-Test: {len(leak_val_test)}")
            success = False
        else:
            logger.info("Deduplication verification PASSED. Zero overlap between train, val, and test splits.")
            
    # 3. Check reports
    reports_dir = config.dataset.reports_dir
    report_files = [
        "validation_report.md",
        "conflict_report.md",
        "cleaning_report.md",
        "dataset_statistics.json",
        "class_distribution.png",
        "sentence_length_distribution.png",
        "top_word_frequencies.png",
        "pipeline_reduction_summary.png"
    ]
    
    for rf in report_files:
        p = os.path.join(reports_dir, rf)
        if not os.path.exists(p):
            logger.error(f"Report file {rf} is missing in reports folder.")
            success = False
            
    if success:
        logger.info("=========================================")
        logger.info("   ALL QUALITY ASSURANCE CHECKS PASSED   ")
        logger.info("=========================================")
    else:
        logger.error("=========================================")
        logger.error("     QUALITY ASSURANCE CHECKS FAILED     ")
        logger.error("=========================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
