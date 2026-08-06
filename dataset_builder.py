import os
import random
import logging
import pandas as pd
from typing import Tuple, List, Dict
import requests
from datasets import load_dataset
from preprocessing import preprocess_text
from config import AppConfig

logger = logging.getLogger(__name__)

class DatasetBuilder:
    def __init__(self, config: AppConfig):
        self.config = config
        self.data_dir = config.dataset.data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.random_seed = config.dataset.random_seed
        random.seed(self.random_seed)

    def download_davidson(self) -> pd.DataFrame:
        """Downloads and maps the Davidson et al. dataset."""
        logger.info("Loading Davidson et al. dataset...")
        url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
        csv_path = os.path.join(self.data_dir, "davidson.csv")
        
        if not os.path.exists(csv_path):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open(csv_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                logger.error(f"Failed to download Davidson dataset: {e}")
                return pd.DataFrame(columns=["text", "label"])
                
        df = pd.read_csv(csv_path)
        # Davidson columns: class, tweet. 
        # class: 0 - hate speech, 1 - offensive language, 2 - neither
        # Map to: 0 - Safe, 1 - Offensive, 2 - Hate Speech
        mapping = {0: 2, 1: 1, 2: 0}
        df_clean = pd.DataFrame({
            "text": df["tweet"],
            "label": df["class"].map(mapping),
            "source": "davidson"
        })
        logger.info(f"Davidson loaded: {len(df_clean)} rows.")
        return df_clean

    def download_olid(self) -> pd.DataFrame:
        """Loads and maps the OLID dataset from HuggingFace."""
        logger.info("Loading OLID dataset...")
        try:
            # OLID Subtask A is available in HF cardiffnlp/tweet_eval under 'offensive'
            # labels: 0 -> non-offensive, 1 -> offensive
            # We map 0 -> 0 (Safe), 1 -> 1 (Offensive). Note OLID doesn't label Hate Speech separately in subtask A.
            dataset = load_dataset("cardiffnlp/tweet_eval", "offensive")
            records = []
            for split in ["train", "validation", "test"]:
                for item in dataset[split]:
                    # map: 0 -> 0 (Safe), 1 -> 1 (Offensive)
                    records.append({
                        "text": item["text"],
                        "label": 0 if item["label"] == 0 else 1,
                        "source": "olid"
                    })
            df = pd.DataFrame(records)
            logger.info(f"OLID loaded: {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Failed to load OLID: {e}. Returning empty DataFrame.")
            return pd.DataFrame(columns=["text", "label", "source"])

    def download_hatexplain(self) -> pd.DataFrame:
        """Loads and maps HateXplain dataset from GitHub/HuggingFace raw data."""
        logger.info("Loading HateXplain dataset...")
        import json
        json_path = os.path.join(self.data_dir, "hatexplain.json")
        
        # Download if not cached
        if not os.path.exists(json_path):
            url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
            try:
                response = requests.get(url, headers=headers, timeout=60)
                response.raise_for_status()
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
            except Exception as e:
                logger.error(f"Failed to download HateXplain JSON: {e}")
                return pd.DataFrame(columns=["text", "label", "source"])
                
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            records = []
            from collections import Counter
            for post_id, item in data.items():
                tokens = item.get("post_tokens", [])
                text = " ".join(tokens)
                
                # Extract labels from annotators
                annotators = item.get("annotators", [])
                if not annotators:
                    continue
                labels = [ann["label"] for ann in annotators]
                
                # Majority vote
                most_common = Counter(labels).most_common(1)[0][0]
                
                # Map HateXplain string labels to unified target labels:
                # "hatespeech" -> 2 (Hate Speech)
                # "offensive" -> 1 (Offensive)
                # "normal" -> 0 (Safe)
                if most_common == "hatespeech":
                    unified_label = 2
                elif most_common == "offensive":
                    unified_label = 1
                else:
                    unified_label = 0
                    
                records.append({
                    "text": text,
                    "label": unified_label,
                    "source": "hatexplain"
                })
                
            df = pd.DataFrame(records)
            logger.info(f"HateXplain loaded: {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Failed to parse HateXplain: {e}. Returning empty DataFrame.")
            return pd.DataFrame(columns=["text", "label", "source"])


    def download_jigsaw(self) -> pd.DataFrame:
        """Loads and maps the Jigsaw Toxic Comment dataset."""
        logger.info("Loading Jigsaw dataset...")
        try:
            # We will load from the community mirror that is direct:
            # 'thesofakillers/jigsaw-toxic-comment-classification-challenge'
            dataset = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge", trust_remote_code=True)
            records = []
            # We use train split and sample from it to save memory/compute
            # Since the raw training set has 159k rows, we'll iterate and sample.
            for item in dataset["train"]:
                text = item["comment_text"]
                # Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
                identity_hate = int(item["identity_hate"])
                threat = int(item["threat"])
                toxic = int(item["toxic"])
                severe_toxic = int(item["severe_toxic"])
                obscene = int(item["obscene"])
                insult = int(item["insult"])
                
                # Mapping logic:
                if identity_hate == 1 or threat == 1:
                    label = 2  # Hate Speech
                elif toxic == 1 or severe_toxic == 1 or obscene == 1 or insult == 1:
                    label = 1  # Offensive
                else:
                    label = 0  # Safe
                    
                records.append({
                    "text": text,
                    "label": label,
                    "source": "jigsaw"
                })
            df = pd.DataFrame(records)
            logger.info(f"Jigsaw loaded: {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Failed to load Jigsaw: {e}. Falling back to GitHub mirror raw CSV.")
            # Fallback to direct download from public github if HuggingFace repository fails
            try:
                mirror_url = "https://raw.githubusercontent.com/trip3ee/Jigsaw-Toxic-Comment-Classification-Challenge/master/data/train.csv"
                csv_path = os.path.join(self.data_dir, "jigsaw_train_mirror.csv")
                if not os.path.exists(csv_path):
                    response = requests.get(mirror_url, timeout=60)
                    response.raise_for_status()
                    with open(csv_path, "wb") as f:
                        f.write(response.content)
                df = pd.read_csv(csv_path)
                records = []
                for _, row in df.iterrows():
                    text = str(row["comment_text"])
                    identity_hate = int(row["identity_hate"])
                    threat = int(row["threat"])
                    toxic = int(row["toxic"])
                    severe_toxic = int(row["severe_toxic"])
                    obscene = int(row["obscene"])
                    insult = int(row["insult"])
                    
                    if identity_hate == 1 or threat == 1:
                        label = 2
                    elif toxic == 1 or severe_toxic == 1 or obscene == 1 or insult == 1:
                        label = 1
                    else:
                        label = 0
                    records.append({"text": text, "label": label, "source": "jigsaw"})
                df_out = pd.DataFrame(records)
                logger.info(f"Jigsaw fallback loaded: {len(df_out)} rows.")
                return df_out
            except Exception as ex:
                logger.error(f"Jigsaw fallback failed: {ex}. Returning empty DataFrame.")
                return pd.DataFrame(columns=["text", "label", "source"])

    def download_civil_comments(self) -> pd.DataFrame:
        """Loads and maps the Civil Comments dataset using streaming to save memory."""
        logger.info("Loading Civil Comments dataset (streaming)...")
        try:
            # google/civil_comments is available in HF. We load it with streaming=True.
            dataset = load_dataset("google/civil_comments", split="train", streaming=True)
            records = []
            
            # Since Civil Comments has 1.8M rows, we iterate and collect a subset
            # We want to collect up to 30,000 samples to keep it balanced and performant
            max_samples = 30000
            count = 0
            for item in dataset:
                if count >= max_samples:
                    break
                text = item["text"]
                
                # Columns are floats in range [0, 1] represent annotator agreement fraction.
                # Thresholding at 0.5:
                identity_attack = item.get("identity_attack", 0.0) or 0.0
                threat = item.get("threat", 0.0) or 0.0
                toxicity = item.get("toxicity", 0.0) or 0.0
                severe_toxicity = item.get("severe_toxicity", 0.0) or 0.0
                obscene = item.get("obscene", 0.0) or 0.0
                insult = item.get("insult", 0.0) or 0.0
                sexual_explicit = item.get("sexual_explicit", 0.0) or 0.0
                
                if identity_attack >= 0.5 or threat >= 0.5:
                    label = 2  # Hate Speech
                elif toxicity >= 0.5 or severe_toxicity >= 0.5 or obscene >= 0.5 or insult >= 0.5 or sexual_explicit >= 0.5:
                    label = 1  # Offensive
                else:
                    label = 0  # Safe
                    
                records.append({
                    "text": text,
                    "label": label,
                    "source": "civil_comments"
                })
                count += 1
                
            df = pd.DataFrame(records)
            logger.info(f"Civil Comments loaded: {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Failed to load Civil Comments: {e}. Returning empty DataFrame.")
            return pd.DataFrame(columns=["text", "label", "source"])

    def build_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Combines datasets, cleans them, deduplicates, balances, 
        and splits into train, val, and test.
        """
        logger.info("Starting dataset build process...")
        
        # Load all datasets
        df_davidson = self.download_davidson()
        df_olid = self.download_olid()
        df_hatexplain = self.download_hatexplain()
        df_jigsaw = self.download_jigsaw()
        df_civil = self.download_civil_comments()
        
        # Combine
        combined_df = pd.concat([df_davidson, df_olid, df_hatexplain, df_jigsaw, df_civil], ignore_index=True)
        logger.info(f"Total raw combined size: {len(combined_df)}")
        
        # Drop duplicates on raw text
        combined_df = combined_df.drop_duplicates(subset=["text"])
        logger.info(f"After initial drop duplicates: {len(combined_df)}")
        
        # Filter empty or null raw text
        combined_df = combined_df[combined_df["text"].astype(str).str.strip() != ""]
        logger.info(f"After removing empty raw text: {len(combined_df)}")
        
        # PRE-SAMPLING OPTIMIZATION
        # To avoid preprocessing hundreds of thousands of samples, we sample a candidate pool first.
        sample_size = self.config.dataset.sample_size_per_class
        if sample_size > 0:
            candidate_pool = pd.DataFrame()
            for label in [0, 1, 2]:
                class_subset = combined_df[combined_df["label"] == label]
                num_available = len(class_subset)
                # We sample 3x the target size to allow for duplicates/empty rows dropped during preprocessing
                target_cand = min(sample_size * 3, num_available)
                if target_cand > 0:
                    sampled_cand = class_subset.sample(n=target_cand, random_state=self.random_seed)
                    candidate_pool = pd.concat([candidate_pool, sampled_cand], ignore_index=True)
            logger.info(f"Optimized: Candidate pool of {len(candidate_pool)} selected for preprocessing (out of {len(combined_df)} raw rows)")
            combined_df = candidate_pool
            
        # Apply preprocessing
        logger.info("Applying preprocessing...")
        prep_cfg = self.config.preprocessing
        
        preprocessed_texts = []
        for idx, row in enumerate(combined_df.itertuples()):
            if idx % 1000 == 0 and idx > 0:
                logger.info(f"Preprocessed {idx} rows...")
            clean = preprocess_text(
                row.text,
                lowercase=prep_cfg.lowercase,
                remove_urls=prep_cfg.remove_urls,
                remove_mentions=prep_cfg.remove_mentions,
                remove_html=prep_cfg.remove_html,
                normalize_emoji=prep_cfg.normalize_emojis,
                normalize_repeated=prep_cfg.normalize_repeated_chars,
                norm_unicode=prep_cfg.normalize_unicode,
                process_hashtags=prep_cfg.handle_hashtags,
                language_filter=prep_cfg.language_filter,
                spell_correct=True
            )
            preprocessed_texts.append(clean)
            
        combined_df["clean_text"] = preprocessed_texts
        
        # Filter empty or noisy preprocessed text
        combined_df = combined_df[combined_df["clean_text"].str.strip() != ""]
        logger.info(f"After removing empty cleaned text: {len(combined_df)}")
        
        # Drop duplicates on clean text
        combined_df = combined_df.drop_duplicates(subset=["clean_text"])
        logger.info(f"After final deduplication on clean text: {len(combined_df)}")
        
        # Remove conflicting labels: same clean_text with different labels
        label_counts = combined_df.groupby("clean_text")["label"].nunique()
        conflicting_texts = label_counts[label_counts > 1].index
        if len(conflicting_texts) > 0:
            logger.info(f"Removing {len(conflicting_texts)} conflicting samples...")
            combined_df = combined_df[~combined_df["clean_text"].isin(conflicting_texts)]
            logger.info(f"After removing conflicting labels: {len(combined_df)}")

        # Evaluate class distributions
        logger.info(f"Class distribution before balancing:\n{combined_df['label'].value_counts()}")
        
        # Class balancing & final sampling
        balanced_df = pd.DataFrame()
        for label in [0, 1, 2]:
            class_subset = combined_df[combined_df["label"] == label]
            num_available = len(class_subset)
            
            if sample_size > 0:
                if num_available >= sample_size:
                    # Undersample to configuration size
                    sampled = class_subset.sample(n=sample_size, random_state=self.random_seed)
                    logger.info(f"Class {label}: Undersampled from {num_available} to {sample_size}")
                else:
                    # Oversample with replacement to reach target
                    sampled = class_subset.sample(n=sample_size, replace=True, random_state=self.random_seed)
                    logger.info(f"Class {label}: Oversampled from {num_available} to {sample_size}")
            else:
                # No sampling requested, use full class subset
                sampled = class_subset
                logger.info(f"Class {label}: Keeping all {num_available} samples")
                
            balanced_df = pd.concat([balanced_df, sampled], ignore_index=True)
            
        # Shuffle final dataset
        balanced_df = balanced_df.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)
        logger.info(f"Balanced dataset size: {len(balanced_df)}")
        logger.info(f"Balanced class distribution:\n{balanced_df['label'].value_counts()}")
        
        # Split into train, val, test
        train_ratio = self.config.dataset.train_ratio
        val_ratio = self.config.dataset.val_ratio
        test_ratio = self.config.dataset.test_ratio
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, "Split ratios must sum to 1.0"
        
        n = len(balanced_df)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)
        
        train_df = balanced_df.iloc[:train_end].copy()
        val_df = balanced_df.iloc[train_end:val_end].copy()
        test_df = balanced_df.iloc[val_end:].copy()
        
        logger.info(f"Splits sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Save CSV files
        train_df.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.data_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.data_dir, "test.csv"), index=False)
        logger.info("Splits saved successfully to CSV.")
        
        return train_df, val_df, test_df

if __name__ == "__main__":
    # Run builder
    from config import load_config
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    
    # Builds the dataset using the sample size specified in configs/config.yaml
    builder = DatasetBuilder(config)
    train, val, test = builder.build_dataset()
    print("Done building dataset!")
