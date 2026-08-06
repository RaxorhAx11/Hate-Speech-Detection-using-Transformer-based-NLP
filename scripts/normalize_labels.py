import os
import sys
import pandas as pd
import json
from collections import Counter

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.config import load_config
from scripts.utils import setup_logging, save_json

logger = setup_logging("normalize_labels")

def normalize_davidson(proc_dir: str) -> pd.DataFrame:
    src_path = os.path.join(proc_dir, "davidson_validated.csv")
    if not os.path.exists(src_path):
        return None
        
    df = pd.read_csv(src_path)
    # class: 0 - hate speech, 1 - offensive language, 2 - neither
    mapping = {0: 2, 1: 1, 2: 0}
    
    normalized_df = pd.DataFrame({
        "text": df["tweet"],
        "label": df["class"].map(mapping),
        "source": "davidson"
    })
    
    dest_path = os.path.join(proc_dir, "davidson_normalized.csv")
    normalized_df.to_csv(dest_path, index=False, encoding="utf-8")
    logger.info(f"Davidson normalized: {len(normalized_df)} rows saved to {dest_path}")
    return normalized_df

def normalize_olid(proc_dir: str) -> pd.DataFrame:
    src_path = os.path.join(proc_dir, "olid_validated.csv")
    if not os.path.exists(src_path):
        return None
        
    df = pd.read_csv(src_path)
    # label: 0 - non-offensive, 1 - offensive
    mapping = {0: 0, 1: 1}
    
    normalized_df = pd.DataFrame({
        "text": df["text"],
        "label": df["label"].map(mapping),
        "source": "olid"
    })
    
    dest_path = os.path.join(proc_dir, "olid_normalized.csv")
    normalized_df.to_csv(dest_path, index=False, encoding="utf-8")
    logger.info(f"OLID normalized: {len(normalized_df)} rows saved to {dest_path}")
    return normalized_df

def normalize_hatexplain(proc_dir: str) -> pd.DataFrame:
    src_path = os.path.join(proc_dir, "hatexplain_validated.json")
    if not os.path.exists(src_path):
        return None
        
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    records = []
    for post_id, val in data.items():
        text = " ".join(val["post_tokens"])
        annotators = val["annotators"]
        labels = [ann["label"] for ann in annotators]
        
        # Majority vote
        counts = Counter(labels)
        most_common = counts.most_common(1)[0][0]
        
        # Handle ties: if all annotators differ, map based on presence of severe labels
        if len(counts) == len(labels):
            if "hatespeech" in counts:
                most_common = "hatespeech"
            elif "offensive" in counts:
                most_common = "offensive"
            else:
                most_common = "normal"
                
        # Map labels
        if most_common == "hatespeech":
            label = 2
        elif most_common == "offensive":
            label = 1
        else:
            label = 0
            
        records.append({
            "text": text,
            "label": label,
            "source": "hatexplain"
        })
        
    normalized_df = pd.DataFrame(records)
    dest_path = os.path.join(proc_dir, "hatexplain_normalized.csv")
    normalized_df.to_csv(dest_path, index=False, encoding="utf-8")
    logger.info(f"HateXplain normalized: {len(normalized_df)} rows saved to {dest_path}")
    return normalized_df

def normalize_jigsaw(proc_dir: str) -> pd.DataFrame:
    src_path = os.path.join(proc_dir, "jigsaw_validated.csv")
    if not os.path.exists(src_path):
        return None
        
    df = pd.read_csv(src_path)
    
    # Mapping rule logic
    def map_row(row):
        identity_hate = int(row["identity_hate"])
        threat = int(row["threat"])
        toxic = int(row["toxic"])
        severe_toxic = int(row["severe_toxic"])
        obscene = int(row["obscene"])
        insult = int(row["insult"])
        
        if identity_hate == 1 or threat == 1:
            return 2
        elif toxic == 1 or severe_toxic == 1 or obscene == 1 or insult == 1:
            return 1
        else:
            return 0
            
    labels = df.apply(map_row, axis=1)
    
    normalized_df = pd.DataFrame({
        "text": df["comment_text"],
        "label": labels,
        "source": "jigsaw"
    })
    
    dest_path = os.path.join(proc_dir, "jigsaw_normalized.csv")
    normalized_df.to_csv(dest_path, index=False, encoding="utf-8")
    logger.info(f"Jigsaw normalized: {len(normalized_df)} rows saved to {dest_path}")
    return normalized_df

def normalize_civil_comments(proc_dir: str) -> pd.DataFrame:
    src_path = os.path.join(proc_dir, "civil_comments_validated.csv")
    if not os.path.exists(src_path):
        return None
        
    df = pd.read_csv(src_path)
    
    # Mapping rule logic
    def map_row(row):
        identity_attack = float(row.get("identity_attack", 0.0) or 0.0)
        threat = float(row.get("threat", 0.0) or 0.0)
        toxicity = float(row.get("toxicity", 0.0) or 0.0)
        severe_toxicity = float(row.get("severe_toxicity", 0.0) or 0.0)
        obscene = float(row.get("obscene", 0.0) or 0.0)
        insult = float(row.get("insult", 0.0) or 0.0)
        sexual_explicit = float(row.get("sexual_explicit", 0.0) or 0.0)
        
        if identity_attack >= 0.5 or threat >= 0.5:
            return 2
        elif toxicity >= 0.5 or severe_toxicity >= 0.5 or obscene >= 0.5 or insult >= 0.5 or sexual_explicit >= 0.5:
            return 1
        else:
            return 0
            
    labels = df.apply(map_row, axis=1)
    
    normalized_df = pd.DataFrame({
        "text": df["text"],
        "label": labels,
        "source": "civil_comments"
    })
    
    dest_path = os.path.join(proc_dir, "civil_comments_normalized.csv")
    normalized_df.to_csv(dest_path, index=False, encoding="utf-8")
    logger.info(f"Civil Comments normalized: {len(normalized_df)} rows saved to {dest_path}")
    return normalized_df

def main():
    config = load_config()
    proc_dir = config.dataset.processed_dir
    os.makedirs(proc_dir, exist_ok=True)
    
    logger.info("--- STARTING LABEL NORMALIZATION ---")
    
    normalize_davidson(proc_dir)
    normalize_olid(proc_dir)
    normalize_hatexplain(proc_dir)
    normalize_jigsaw(proc_dir)
    normalize_civil_comments(proc_dir)
    
    # Save mapping details to JSON
    mapping_details = {
        "davidson": {
            "original_labels": {
                "0": "hate speech",
                "1": "offensive language",
                "2": "neither"
            },
            "mapping": {"0": 2, "1": 1, "2": 0}
        },
        "olid": {
            "original_labels": {
                "0": "non-offensive",
                "1": "offensive"
            },
            "mapping": {"0": 0, "1": 1}
        },
        "hatexplain": {
            "original_labels": {
                "normal": "Safe",
                "offensive": "Offensive",
                "hatespeech": "Hate Speech"
            },
            "mapping": {"normal": 0, "offensive": 1, "hatespeech": 2}
        },
        "jigsaw": {
            "description": "Multi-label rule mapping based on binary flags.",
            "rules": {
                "Hate Speech (2)": "identity_hate == 1 or threat == 1",
                "Offensive (1)": "toxic == 1 or severe_toxic == 1 or obscene == 1 or insult == 1",
                "Safe (0)": "All other cases"
            }
        },
        "civil_comments": {
            "description": "Annotator fraction rule mapping using 0.5 threshold.",
            "rules": {
                "Hate Speech (2)": "identity_attack >= 0.5 or threat >= 0.5",
                "Offensive (1)": "toxicity >= 0.5 or severe_toxicity >= 0.5 or obscene >= 0.5 or insult >= 0.5 or sexual_explicit >= 0.5",
                "Safe (0)": "All other cases"
            }
        },
        "target_labels": {
            "0": "Safe",
            "1": "Offensive",
            "2": "Hate Speech"
        }
    }
    
    mapping_json_path = os.path.join(proc_dir, "label_mapping.json")
    save_json(mapping_details, mapping_json_path)
    logger.info(f"Label mapping configuration saved to {mapping_json_path}")
    logger.info("--- LABEL NORMALIZATION COMPLETED ---")

if __name__ == "__main__":
    main()
