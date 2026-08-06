import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.config import load_config
from scripts.utils import setup_logging, save_json

logger = setup_logging("split_dataset")

def check_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Verifies that there is zero text overlap between train, val, and test splits."""
    train_texts = set(train_df["text"].astype(str).tolist())
    val_texts = set(val_df["text"].astype(str).tolist())
    test_texts = set(test_df["text"].astype(str).tolist())
    
    leak_train_val = train_texts.intersection(val_texts)
    leak_train_test = train_texts.intersection(test_texts)
    leak_val_test = val_texts.intersection(test_texts)
    
    has_leakage = len(leak_train_val) > 0 or len(leak_train_test) > 0 or len(leak_val_test) > 0
    if has_leakage:
        logger.error(f"DATA LEAKAGE DETECTED! Train-Val overlap: {len(leak_train_val)}, Train-Test overlap: {len(leak_train_test)}, Val-Test overlap: {len(leak_val_test)}")
    else:
        logger.info("Leakage verification passed: 0 text overlaps between train, validation, and test splits.")
        
    return not has_leakage

def calculate_class_weights(df: pd.DataFrame) -> dict:
    """Calculates class weights for loss balancing."""
    counts = df["label"].value_counts().to_dict()
    total = sum(counts.values())
    num_classes = len(counts)
    
    # Standard formula: weight = total / (num_classes * class_count)
    weights = {}
    for label, count in counts.items():
        weights[int(label)] = round(total / (num_classes * count), 4) if count > 0 else 0.0
        
    return weights

def main():
    config = load_config()
    merged_dir = config.dataset.merged_dir
    data_dir = config.dataset.data_dir
    
    src_path = os.path.join(merged_dir, "clean_dataset.csv")
    if not os.path.exists(src_path):
        logger.error(f"Merged dataset file {src_path} not found.")
        return
        
    df = pd.read_csv(src_path)
    # Ensure text is string and drop any na
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    
    logger.info(f"Loaded merged dataset: {len(df)} rows.")
    
    # 1. Generate Balancing Recommendations
    counts = df["label"].value_counts().to_dict()
    class_names = {0: "Safe", 1: "Offensive", 2: "Hate Speech"}
    
    logger.info("Class distribution:")
    for k, v in counts.items():
        logger.info(f"Class {k} ({class_names[k]}): {v} samples ({v/len(df)*100:.2f}%)")
        
    class_weights = calculate_class_weights(df)
    logger.info(f"Calculated class weights: {class_weights}")
    
    # 2. Check for Subsampling config (undersampling only, no aggressive duplication)
    sample_size = config.dataset.sample_size_per_class
    if sample_size > 0:
        logger.info(f"Subsampling requested: target size per class = {sample_size}")
        sampled_dfs = []
        for label in [0, 1, 2]:
            class_df = df[df["label"] == label]
            if len(class_df) > sample_size:
                # Undersample to prevent imbalance
                sampled_class_df = class_df.sample(n=sample_size, random_state=config.dataset.random_seed)
                logger.info(f"Class {label} ({class_names[label]}): Undersampled from {len(class_df)} to {sample_size}")
            else:
                # Do NOT duplicate/oversample, just use all available
                sampled_class_df = class_df
                logger.info(f"Class {label} ({class_names[label]}): Keeping all {len(class_df)} samples (insufficient data to reach target {sample_size})")
            sampled_dfs.append(sampled_class_df)
        df = pd.concat(sampled_dfs, ignore_index=True)
        logger.info(f"Total size after config-based subsampling: {len(df)} rows.")
        
    # 3. Stratified Splitting
    train_ratio = config.dataset.train_ratio
    val_ratio = config.dataset.val_ratio
    test_ratio = config.dataset.test_ratio
    
    # Normalise ratios if they don't sum to 1.0 perfectly
    total_ratio = train_ratio + val_ratio + test_ratio
    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio
    
    # Split train and remaining (val + test)
    remaining_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=remaining_ratio,
        random_state=config.dataset.random_seed,
        stratify=df["label"]
    )
    
    # Split remaining into val and test
    val_test_ratio = val_ratio / remaining_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1.0 - val_test_ratio,
        random_state=config.dataset.random_seed,
        stratify=temp_df["label"]
    )
    
    logger.info(f"Splits generated: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 4. Verification Check for leakage
    check_leakage(train_df, val_df, test_df)
    
    # 5. Export Output files
    # Save train.csv, validation.csv, test.csv in the dataset directory
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    logger.info(f"Saved {train_path}")
    logger.info(f"Saved {val_path}")
    logger.info(f"Saved {test_path}")
    
    # Save balancing and splitting metadata
    split_metadata = {
        "class_counts_merged": counts,
        "class_weights_loss": class_weights,
        "splits": {
            "train": {
                "total": len(train_df),
                "label_distribution": train_df["label"].value_counts().to_dict()
            },
            "validation": {
                "total": len(val_df),
                "label_distribution": val_df["label"].value_counts().to_dict()
            },
            "test": {
                "total": len(test_df),
                "label_distribution": test_df["label"].value_counts().to_dict()
            }
        }
    }
    
    save_json(split_metadata, os.path.join(merged_dir, "split_metadata.json"))
    logger.info("--- DATASET SPLITTING COMPLETED ---")

if __name__ == "__main__":
    main()
