import os
import sys
import pandas as pd
import numpy as np
import json
import re
import shutil
from collections import Counter
from tqdm import tqdm

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.config import load_config
from scripts.utils import setup_logging, save_markdown, save_json

logger = setup_logging("merge_datasets")

def remove_noise(df: pd.DataFrame, text_col: str, min_length: int) -> tuple:
    """Removes short, numeric, symbol-only, and spam text samples."""
    total_in = len(df)
    
    # 1. Null/NaN check
    df = df.dropna(subset=[text_col])
    
    # 2. Too short text check
    is_too_short = df[text_col].astype(str).str.len() < min_length
    is_too_few_words = df[text_col].astype(str).apply(lambda t: len(str(t).split())) < 2
    
    # 3. Only numbers check
    is_only_numbers = df[text_col].astype(str).str.match(r'^\d+$')
    
    # 4. Only symbols check (no alphanumeric characters)
    is_only_symbols = df[text_col].astype(str).str.match(r'^[^a-zA-Z0-9]+$')
    
    # 5. Repeated word spam check (lexical diversity check for longer text)
    def is_spam_text(text: str) -> bool:
        words = str(text).split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.2:  # Same word repeated over and over
                return True
        return False
        
    is_spam = df[text_col].astype(str).apply(is_spam_text)
    
    noise_mask = is_too_short | is_too_few_words | is_only_numbers | is_only_symbols | is_spam
    noise_df = df[noise_mask]
    clean_df = df[~noise_mask]
    
    noise_stats = {
        "too_short": int(is_too_short.sum()),
        "too_few_words": int(is_too_few_words.sum()),
        "only_numbers": int(is_only_numbers.sum()),
        "only_symbols": int(is_only_symbols.sum()),
        "spam": int(is_spam.sum()),
        "total_removed": total_in - len(clean_df)
    }
    
    return clean_df, noise_stats

def detect_near_duplicates(df: pd.DataFrame, text_col: str, threshold: float) -> list:
    """Detects near duplicate indices using prefix-filtered inverted index and Jaccard similarity."""
    logger.info("Tokenizing texts for near-duplicate checks...")
    texts = df[text_col].astype(str).tolist()
    word_sets = [set(t.split()) for t in texts]
    lengths = [len(ws) for ws in word_sets]
    
    # 1. Count word frequencies across the corpus to rank words
    logger.info("Computing corpus word frequencies...")
    all_words = []
    for ws in word_sets:
        all_words.extend(ws)
    word_counts = Counter(all_words)
    
    # 2. Build inverted index for prefix words only
    logger.info("Building prefix-filtered inverted index...")
    inverted_index = {}
    sorted_word_lists = []
    
    for idx, ws in enumerate(word_sets):
        # Sort words in this document by their global corpus count (increasing, i.e. least frequent first)
        sorted_words = sorted(list(ws), key=lambda w: word_counts[w])
        sorted_word_lists.append(sorted_words)
        
        l_i = len(ws)
        # Prefix length k = int(l_i * (1 - threshold)) + 1
        k = int(l_i * (1.0 - threshold)) + 1
        
        # Index only the first k words (the prefix)
        prefix_words = sorted_words[:k]
        for word in prefix_words:
            if word not in inverted_index:
                inverted_index[word] = []
            inverted_index[word].append(idx)
            
    duplicates_to_remove = set()
    n = len(df)
    
    logger.info("Running prefix-filtered Jaccard similarity checks...")
    for i in tqdm(range(n), desc="Near duplicate check"):
        if i in duplicates_to_remove:
            continue
        ws_i = word_sets[i]
        len_i = lengths[i]
        if len_i == 0:
            continue
            
        sorted_words_i = sorted_word_lists[i]
        k_i = int(len_i * (1.0 - threshold)) + 1
        prefix_words_i = sorted_words_i[:k_i]
        
        # Query candidates using only prefix words
        candidates = set()
        for word in prefix_words_i:
            candidates.update(inverted_index.get(word, []))
            
        # Filter candidates: only check indices greater than i, and not already scheduled for removal
        candidates = {j for j in candidates if j > i and j not in duplicates_to_remove}
        
        for j in candidates:
            len_j = lengths[j]
            
            # Length ratio check
            min_len, max_len = min(len_i, len_j), max(len_i, len_j)
            if min_len / max_len < threshold:
                continue
                
            ws_j = word_sets[j]
            intersection = len(ws_i.intersection(ws_j))
            union = len_i + len_j - intersection
            sim = intersection / union if union > 0 else 0
            
            if sim >= threshold:
                duplicates_to_remove.add(j)
                
    return list(duplicates_to_remove)

def resolve_conflicts(df: pd.DataFrame, text_col: str, strategy: str) -> tuple:
    """Detects conflicting labels for identical text and resolves them."""
    # Find texts with multiple labels
    grouped = df.groupby(text_col)["label"].nunique()
    conflicting_texts = grouped[grouped > 1].index.tolist()
    
    resolved_records = []
    conflict_details = []
    
    # Partition df into conflicting and non-conflicting
    non_conflict_df = df[~df[text_col].isin(conflicting_texts)]
    conflict_df = df[df[text_col].isin(conflicting_texts)]
    
    for text in conflicting_texts:
        sub_df = conflict_df[conflict_df[text_col] == text]
        labels = sub_df["label"].tolist()
        sources = sub_df["source"].tolist()
        
        counts = Counter(labels)
        most_common_label, freq = counts.most_common(1)[0]
        
        # Apply resolution strategy
        resolved_label = None
        if strategy == "majority_vote":
            # If there's a tie, use the severity rule
            if len(counts) > 1 and list(counts.values()).count(freq) > 1:
                # Severity priority tie breaker: 2 > 1 > 0
                resolved_label = max(labels)
            else:
                resolved_label = most_common_label
        elif strategy == "severity_priority":
            resolved_label = max(labels)
        elif strategy == "remove":
            resolved_label = None  # Drop sample completely
            
        if resolved_label is not None:
            # Reconstruct record, using the most common source or a combined source tag
            resolved_records.append({
                "text": text,
                "label": resolved_label,
                "source": "|".join(sorted(list(set(sources))))
            })
            
        conflict_details.append({
            "text": text,
            "original_labels": dict(counts),
            "sources": sources,
            "resolved_label": resolved_label,
            "strategy_used": strategy
        })
        
    resolved_conflict_df = pd.DataFrame(resolved_records)
    final_df = pd.concat([non_conflict_df, resolved_conflict_df], ignore_index=True)
    
    return final_df, conflict_details

def main():
    config = load_config()
    proc_dir = config.dataset.processed_dir
    merged_dir = config.dataset.merged_dir
    reports_dir = config.dataset.reports_dir
    
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    logger.info("--- STARTING DATASET MERGING AND DEDUPLICATION ---")
    
    datasets = ["davidson", "olid", "hatexplain", "jigsaw", "civil_comments"]
    dfs = []
    
    for d in datasets:
        cleaned_path = os.path.join(proc_dir, f"{d}_cleaned.csv")
        if os.path.exists(cleaned_path):
            df = pd.read_csv(cleaned_path)
            dfs.append(df)
            logger.info(f"Loaded {d}_cleaned.csv: {len(df)} rows.")
            
    if not dfs:
        logger.error("No cleaned datasets found to merge.")
        return
        
    # Combine datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    total_merged = len(combined_df)
    logger.info(f"Combined dataset size: {total_merged} rows.")
    
    # 1. Noise Removal
    logger.info("Performing noise removal...")
    cleaned_df, noise_stats = remove_noise(combined_df, "text", config.dataset.min_text_length)
    logger.info(f"Noise removal complete. Removed {noise_stats['total_removed']} rows.")
    
    # 2. Case and Whitespace Duplicate Removal
    logger.info("Deduplicating on exact, case-insensitive, and whitespace matches...")
    # Add temporary canonical column for fast initial deduplication
    cleaned_df["canonical_text"] = cleaned_df["text"].str.lower().str.strip()
    
    total_before_dedup = len(cleaned_df)
    exact_whitespace_removed = total_before_dedup - len(cleaned_df)
    
    # We resolve conflicts BEFORE deduplicating because conflicts represent duplicates with DIFFERENT labels
    # 3. Label Conflict Detection and Resolution
    logger.info("Detecting label conflicts...")
    resolved_df, conflict_details = resolve_conflicts(cleaned_df, "canonical_text", config.dataset.conflict_resolution)
    logger.info(f"Conflict resolution complete. Resolved conflicts for {len(conflict_details)} texts.")
    
    # Deduplicate remaining canonical texts to make sure no duplicate remains
    resolved_df = resolved_df.drop_duplicates(subset=["canonical_text"])
    
    # Reset index BEFORE running near-duplicate check so position indexes align with index labels
    resolved_df = resolved_df.reset_index(drop=True)
    
    # Clean up canonical text column and rename back to normal columns
    resolved_df = resolved_df.rename(columns={"text": "raw_text_ref", "canonical_text": "text"})
    resolved_df = resolved_df[["text", "label", "source"]]
    
    # 4. Near-Duplicate Detection and Removal
    logger.info("Detecting near-duplicates using length-filtered Jaccard similarity...")
    near_dup_indices = detect_near_duplicates(resolved_df, "text", config.dataset.near_duplicate_threshold)
    logger.info(f"Near-duplicate detection complete. Found {len(near_dup_indices)} near-duplicates.")
    
    final_df = resolved_df.drop(index=near_dup_indices).reset_index(drop=True)
    logger.info(f"Final merged dataset size: {len(final_df)} rows.")
    
    # Save the final merged dataset
    dest_path = os.path.join(merged_dir, "clean_dataset.csv")
    final_df.to_csv(dest_path, index=False, encoding="utf-8")
    
    # Copy to dataset root for downstream model compatibility
    shutil_dest_path = os.path.join(config.dataset.data_dir, "clean_dataset.csv")
    shutil.copy(dest_path, shutil_dest_path)
    logger.info(f"Saved merged dataset to {dest_path} and copied to {shutil_dest_path}")
    
    # Write Conflict Report
    conflict_report = "# Label Conflict Resolution Report\n\n"
    conflict_report += f"**Resolution Strategy Configuration**: `{config.dataset.conflict_resolution}`\n\n"
    conflict_report += f"Total conflicting text groups resolved: **{len(conflict_details)}**\n\n"
    
    conflict_report += "### Sample Conflicts and Resolutions\n\n"
    conflict_report += "| Text | Label Frequencies | Source Datasets | Resolved Label |\n"
    conflict_report += "| :--- | :--- | :--- | :--- |\n"
    
    # List top 20 conflicts for readability
    for c in conflict_details[:20]:
        original_labels_str = ", ".join([f"class {k}: {v} times" for k, v in c["original_labels"].items()])
        sources_str = ", ".join(set(c["sources"]))
        resolved_label_str = f"class {c['resolved_label']}" if c['resolved_label'] is not None else "DROPPED"
        text_truncated = c['text'][:60] + "..." if len(c['text']) > 60 else c['text']
        conflict_report += f"| \"{text_truncated}\" | {original_labels_str} | {sources_str} | **{resolved_label_str}** |\n"
        
    save_markdown(conflict_report, os.path.join(reports_dir, "conflict_report.md"))
    logger.info(f"Conflict report saved to {os.path.join(reports_dir, 'conflict_report.md')}")
    
    # Save statistics metadata
    merge_stats = {
        "total_merged_input": total_merged,
        "noise_removed": noise_stats,
        "exact_and_canonical_duplicates_removed": exact_whitespace_removed,
        "conflicts_detected": len(conflict_details),
        "near_duplicates_removed": len(near_dup_indices),
        "total_output_rows": len(final_df)
    }
    
    save_json(merge_stats, os.path.join(merged_dir, "merge_statistics.json"))
    logger.info("--- DATASET MERGING AND DEDUPLICATION COMPLETED ---")

if __name__ == "__main__":
    main()
