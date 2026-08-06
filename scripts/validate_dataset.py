import os
import sys
import pandas as pd
import json
import logging

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.config import load_config
from scripts.utils import setup_logging, save_markdown

logger = setup_logging("validate_dataset")

def check_encoding(file_path: str) -> str:
    """Helper to detect if file is valid UTF-8, else fallback to latin-1."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            f.read(10000)
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"

def validate_dataframe(df: pd.DataFrame, text_col: str, label_cols: list, id_col: str = None) -> dict:
    """Performs validation checks on a pandas DataFrame."""
    stats = {}
    total_rows = len(df)
    stats["total_raw_rows"] = total_rows
    
    # 1. Check missing columns
    missing_cols = [col for col in [text_col] + label_cols if col not in df.columns]
    stats["missing_columns"] = missing_cols
    
    # Drop rows if text column is missing (corrupted)
    if text_col not in df.columns:
        logger.error(f"Text column '{text_col}' missing. Cannot validate.")
        return stats
        
    # 2. Check null/empty text samples
    null_text = df[text_col].isnull().sum()
    empty_text = (df[text_col].astype(str).str.strip() == "").sum()
    stats["null_text_samples"] = int(null_text)
    stats["empty_text_samples"] = int(empty_text)
    
    # 3. Check missing labels
    stats["missing_labels"] = {col: int(df[col].isnull().sum()) for col in label_cols}
    
    # 4. Check duplicate IDs
    if id_col and id_col in df.columns:
        stats["duplicate_ids"] = int(df[id_col].duplicated().sum())
    else:
        stats["duplicate_ids"] = 0
        
    # 5. Check duplicate text samples (exact raw match)
    stats["duplicate_texts"] = int(df[text_col].duplicated().sum())
    
    # Clean step: remove null/empty text, missing labels, and keep first of duplicate text
    clean_df = df.dropna(subset=[text_col] + label_cols)
    clean_df = clean_df[clean_df[text_col].astype(str).str.strip() != ""]
    clean_df = clean_df.drop_duplicates(subset=[text_col])
    
    stats["total_clean_rows"] = len(clean_df)
    stats["removed_corrupted_or_invalid_rows"] = total_rows - len(clean_df)
    
    return stats, clean_df

def validate_davidson(raw_dir: str, proc_dir: str) -> dict:
    file_path = os.path.join(raw_dir, "davidson", "davidson.csv")
    if not os.path.exists(file_path):
        logger.warning("Davidson dataset file not found.")
        return None
        
    encoding = check_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    stats, clean_df = validate_dataframe(df, text_col="tweet", label_cols=["class"])
    stats["encoding"] = encoding
    stats["dataset_name"] = "Davidson et al."
    
    # Save validated file
    dest_path = os.path.join(proc_dir, "davidson_validated.csv")
    clean_df.to_csv(dest_path, index=False, encoding="utf-8")
    return stats

def validate_olid(raw_dir: str, proc_dir: str) -> dict:
    file_path = os.path.join(raw_dir, "olid", "olid.csv")
    if not os.path.exists(file_path):
        logger.warning("OLID dataset file not found.")
        return None
        
    encoding = check_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    stats, clean_df = validate_dataframe(df, text_col="text", label_cols=["label"])
    stats["encoding"] = encoding
    stats["dataset_name"] = "OLID"
    
    dest_path = os.path.join(proc_dir, "olid_validated.csv")
    clean_df.to_csv(dest_path, index=False, encoding="utf-8")
    return stats

def validate_jigsaw(raw_dir: str, proc_dir: str) -> dict:
    file_path = os.path.join(raw_dir, "jigsaw", "train.csv")
    if not os.path.exists(file_path):
        logger.warning("Jigsaw dataset file not found.")
        return None
        
    encoding = check_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    stats, clean_df = validate_dataframe(df, text_col="comment_text", label_cols=label_cols, id_col="id")
    stats["encoding"] = encoding
    stats["dataset_name"] = "Jigsaw Toxic Comment"
    
    dest_path = os.path.join(proc_dir, "jigsaw_validated.csv")
    clean_df.to_csv(dest_path, index=False, encoding="utf-8")
    return stats

def validate_civil_comments(raw_dir: str, proc_dir: str) -> dict:
    file_path = os.path.join(raw_dir, "civil_comments", "train.csv")
    if not os.path.exists(file_path):
        logger.warning("Civil Comments dataset file not found.")
        return None
        
    encoding = check_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    label_cols = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
    stats, clean_df = validate_dataframe(df, text_col="text", label_cols=label_cols)
    stats["encoding"] = encoding
    stats["dataset_name"] = "Civil Comments"
    
    dest_path = os.path.join(proc_dir, "civil_comments_validated.csv")
    clean_df.to_csv(dest_path, index=False, encoding="utf-8")
    return stats

def validate_hatexplain(raw_dir: str, proc_dir: str) -> dict:
    file_path = os.path.join(raw_dir, "hatexplain", "hatexplain.json")
    if not os.path.exists(file_path):
        logger.warning("HateXplain dataset file not found.")
        return None
        
    encoding = check_encoding(file_path)
    with open(file_path, "r", encoding=encoding) as f:
        data = json.load(f)
        
    # Check JSON schema and remove corrupted items
    total_raw = len(data)
    corrupted = 0
    empty = 0
    duplicate_ids = 0
    ids_seen = set()
    clean_data = {}
    
    for key, val in data.items():
        if key in ids_seen:
            duplicate_ids += 1
            continue
        ids_seen.add(key)
        
        # Check structure
        post_tokens = val.get("post_tokens", [])
        annotators = val.get("annotators", [])
        
        if not isinstance(post_tokens, list) or len(post_tokens) == 0:
            corrupted += 1
            continue
            
        text = " ".join(post_tokens).strip()
        if not text:
            empty += 1
            continue
            
        if not isinstance(annotators, list) or len(annotators) == 0:
            corrupted += 1
            continue
            
        # Ensure at least one annotator has a label
        has_label = all("label" in ann for ann in annotators)
        if not has_label:
            corrupted += 1
            continue
            
        clean_data[key] = val
        
    stats = {
        "dataset_name": "HateXplain",
        "encoding": encoding,
        "total_raw_rows": total_raw,
        "missing_columns": [],
        "null_text_samples": 0,
        "empty_text_samples": empty,
        "missing_labels": {"label": 0},
        "duplicate_ids": duplicate_ids,
        "duplicate_texts": 0,
        "total_clean_rows": len(clean_data),
        "removed_corrupted_or_invalid_rows": corrupted + empty + duplicate_ids
    }
    
    dest_path = os.path.join(proc_dir, "hatexplain_validated.json")
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)
        
    return stats

def generate_report(all_stats: list, reports_dir: str):
    """Generates a markdown validation report."""
    md = "# Dataset Validation Report\n\n"
    md += "This report summarizes the data quality and structure validation for each raw dataset before normalization and merging.\n\n"
    
    md += "| Dataset Name | Encoding | Raw Size | Cleaned Size | Corrupted/Invalid Removed | Duplicate Texts (Raw) | Empty Samples |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for stat in all_stats:
        if not stat:
            continue
        md += f"| **{stat['dataset_name']}** | {stat['encoding']} | {stat['total_raw_rows']:,} | {stat['total_clean_rows']:,} | {stat['removed_corrupted_or_invalid_rows']:,} | {stat['duplicate_texts']:,} | {stat['empty_text_samples']:,} |\n"
        
    md += "\n## Key Findings & Actions\n"
    for stat in all_stats:
        if not stat:
            continue
        md += f"\n### {stat['dataset_name']}\n"
        md += f"- **Encoding detected**: `{stat['encoding']}`\n"
        if stat['missing_columns']:
            md += f"- **Missing columns detected**: {stat['missing_columns']}\n"
        else:
            md += f"- **Columns verified**: OK\n"
        md += f"- **Duplicate IDs detected**: {stat['duplicate_ids']:,}\n"
        md += f"- **Empty text rows removed**: {stat['empty_text_samples']:,}\n"
        md += f"- **Invalid label rows removed**: {sum(stat['missing_labels'].values()) if isinstance(stat['missing_labels'], dict) else 0:,}\n"
        
    save_markdown(md, os.path.join(reports_dir, "validation_report.md"))
    logger.info(f"Validation report saved to {os.path.join(reports_dir, 'validation_report.md')}")

def main():
    config = load_config()
    raw_dir = config.dataset.raw_dir
    proc_dir = config.dataset.processed_dir
    reports_dir = config.dataset.reports_dir
    
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger.info("--- STARTING DATASET VALIDATION PIPELINE ---")
    all_stats = []
    
    stats_davidson = validate_davidson(raw_dir, proc_dir)
    stats_hatexplain = validate_hatexplain(raw_dir, proc_dir)
    stats_olid = validate_olid(raw_dir, proc_dir)
    stats_jigsaw = validate_jigsaw(raw_dir, proc_dir)
    stats_civil = validate_civil_comments(raw_dir, proc_dir)
    
    all_stats.extend([stats_davidson, stats_hatexplain, stats_olid, stats_jigsaw, stats_civil])
    
    # Filter None stats (if some files are missing)
    all_stats = [s for s in all_stats if s is not None]
    
    if all_stats:
        generate_report(all_stats, reports_dir)
    else:
        logger.error("No datasets could be validated because all files were missing.")
    
    logger.info("--- DATASET VALIDATION PIPELINE COMPLETED ---")

if __name__ == "__main__":
    main()
