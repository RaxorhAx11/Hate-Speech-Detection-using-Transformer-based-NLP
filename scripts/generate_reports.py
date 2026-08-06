import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.config import load_config
from scripts.utils import setup_logging, save_json

logger = setup_logging("generate_reports")

def generate_stats_and_charts():
    config = load_config()
    data_dir = config.dataset.data_dir
    merged_dir = config.dataset.merged_dir
    reports_dir = config.dataset.reports_dir
    
    # Check if final merged dataset exists
    merged_csv = os.path.join(merged_dir, "clean_dataset.csv")
    if not os.path.exists(merged_csv):
        logger.error(f"Clean merged dataset not found at {merged_csv}")
        return
        
    df = pd.read_csv(merged_csv)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    
    total_samples = len(df)
    logger.info(f"Generating reports for {total_samples} samples...")
    
    # 1. Class Distribution
    class_counts = df["label"].value_counts().to_dict()
    class_names = {0: "Safe", 1: "Offensive", 2: "Hate Speech"}
    class_dist = {class_names.get(int(k), f"Class {k}"): int(v) for k, v in class_counts.items()}
    
    # 2. Sentence Length Statistics (Word count & Char count)
    word_lengths = df["text"].apply(lambda t: len(str(t).split()))
    char_lengths = df["text"].apply(len)
    
    avg_words = float(word_lengths.mean())
    max_words = int(word_lengths.max())
    avg_chars = float(char_lengths.mean())
    max_chars = int(char_lengths.max())
    
    # 3. Vocabulary Size and Top words
    all_words = []
    for text in df["text"]:
        all_words.extend(str(text).split())
        
    unique_words = set(all_words)
    vocab_size = len(unique_words)
    
    word_counts = Counter(all_words)
    top_words = [{"word": w, "count": c} for w, c in word_counts.most_common(30)]
    
    # 4. Read other statistics JSONs
    merge_stats_path = os.path.join(merged_dir, "merge_statistics.json")
    merge_stats = {}
    if os.path.exists(merge_stats_path):
        with open(merge_stats_path, "r", encoding="utf-8") as f:
            merge_stats = json.load(f)
            
    split_stats_path = os.path.join(merged_dir, "split_metadata.json")
    split_stats = {}
    if os.path.exists(split_stats_path):
        with open(split_stats_path, "r", encoding="utf-8") as f:
            split_stats = json.load(f)
            
    # 5. Compile dataset_statistics.json
    stats_data = {
        "total_samples": total_samples,
        "class_distribution": class_dist,
        "sentence_length": {
            "average_words": round(avg_words, 2),
            "max_words": max_words,
            "average_characters": round(avg_chars, 2),
            "max_characters": max_chars
        },
        "vocabulary_size": vocab_size,
        "top_frequent_words": top_words[:20],
        "merge_statistics": merge_stats,
        "split_statistics": split_stats
    }
    
    # Save dataset_statistics.json in both reports/ and dataset/ directories
    save_json(stats_data, os.path.join(reports_dir, "dataset_statistics.json"))
    save_json(stats_data, os.path.join(data_dir, "dataset_statistics.json"))
    logger.info("Saved dataset_statistics.json")
    
    # Set seaborn style for rich aesthetics
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 150
    
    # Plot 1: Class Distribution Chart
    plt.figure(figsize=(8, 5))
    labels = list(class_dist.keys())
    sizes = list(class_dist.values())
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]  # Sleek green, orange, red colors
    
    bars = plt.bar(labels, sizes, color=colors, edgecolor="none", width=0.6)
    plt.title("Unified Hate Speech Dataset Class Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.xlabel("Category", fontsize=12)
    
    # Annotate bar heights
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (total_samples*0.01), f"{height:,}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "class_distribution.png"), dpi=200)
    plt.close()
    logger.info("Saved class_distribution.png")
    
    # Plot 2: Sentence Length Histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(word_lengths, bins=40, kde=True, color="#3498db")
    plt.title("Distribution of Sentence Length (Cleaned Word Count)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Number of Words per Comment", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(avg_words, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Average: {avg_words:.1f} words")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "sentence_length_distribution.png"), dpi=200)
    plt.close()
    logger.info("Saved sentence_length_distribution.png")
    
    # Plot 3: Top Word Frequencies Chart
    plt.figure(figsize=(10, 6))
    top_words_df = pd.DataFrame(top_words[:20])
    sns.barplot(x="count", y="word", data=top_words_df, palette="viridis")
    plt.title("Top 20 Most Frequent Words in Unified Dataset", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Occurrences", fontsize=12)
    plt.ylabel("Word", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "top_word_frequencies.png"), dpi=200)
    plt.close()
    logger.info("Saved top_word_frequencies.png")
    
    # Plot 4: Pipeline Data Reduction / Deduplication Chart (Horizontal Stacked Bar)
    if merge_stats:
        plt.figure(figsize=(10, 4))
        raw_size = merge_stats.get("total_merged_input", total_samples)
        noise = merge_stats.get("noise_removed", {}).get("total_removed", 0)
        dups = merge_stats.get("exact_and_canonical_duplicates_removed", 0)
        near_dups = merge_stats.get("near_duplicates_removed", 0)
        clean_size = merge_stats.get("total_output_rows", total_samples)
        
        stages = ["Raw Merged", "After Noise Clean", "After Exact Deduplication", "After Jaccard Near Deduplication"]
        sizes_stages = [raw_size, raw_size - noise, raw_size - noise - dups, clean_size]
        
        sns.barplot(x=sizes_stages, y=stages, palette="magma")
        plt.title("Data Volume Across Pipeline Stages", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Number of Rows", fontsize=12)
        
        # Add labels on the bar
        for i, val in enumerate(sizes_stages):
            plt.text(val - (raw_size * 0.08), i, f"{val:,}", va='center', ha='right', color='white', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "pipeline_reduction_summary.png"), dpi=200)
        plt.close()
        logger.info("Saved pipeline_reduction_summary.png")

def main():
    logger.info("--- STARTING REPORT GENERATION AND VISUALIZATION ---")
    generate_stats_and_charts()
    logger.info("--- REPORT GENERATION AND VISUALIZATION COMPLETED ---")

if __name__ == "__main__":
    main()
