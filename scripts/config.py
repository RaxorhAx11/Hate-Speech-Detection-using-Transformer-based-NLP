import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 3
    max_length: int = 128
    dropout: float = 0.1

@dataclass
class DatasetConfig:
    data_dir: str = "dataset"
    raw_dir: str = "dataset/raw"
    processed_dir: str = "dataset/processed"
    merged_dir: str = "dataset/merged"
    reports_dir: str = "dataset/reports"
    cache_dir: str = "dataset/cache"
    sample_size_per_class: int = 5000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    min_text_length: int = 5
    conflict_resolution: str = "majority_vote"
    near_duplicate_threshold: float = 0.90

@dataclass
class PreprocessingConfig:
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_html: bool = True
    normalize_emojis: bool = True
    normalize_repeated_chars: bool = True
    normalize_unicode: bool = True
    remove_duplicates: bool = True
    language_filter: str = "en"
    handle_hashtags: bool = True
    remove_emails: bool = True
    normalize_whitespace: bool = True
    normalize_punctuation: bool = True
    remove_invisible_chars: bool = True
    normalize_quotes: bool = True
    handle_reddit_formatting: bool = True
    handle_markdown_formatting: bool = True
    remove_corrupted_unicode: bool = True

@dataclass
class TrainingConfig:
    output_dir: str = "saved_models"
    logging_dir: str = "logs"
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    fp16: bool = False
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    metric_for_best_model: str = "macro_f1"

@dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8000

@dataclass
class VoiceConfig:
    speech_rate: int = 150
    volume: float = 1.0

@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)

def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """Loads configuration from YAML file and returns AppConfig instance."""
    # Find config relative to project root
    if not os.path.isabs(config_path):
        # Check if we are running from inside scripts/ directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        possible_path = os.path.join(parent_dir, config_path)
        if os.path.exists(possible_path):
            config_path = possible_path
        elif os.path.exists(config_path):
            pass
        else:
            # Try workspace root default
            config_path = os.path.join(os.getcwd(), config_path)

    if not os.path.exists(config_path):
        # Return default config if file doesn't exist
        return AppConfig()
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        
    model_data = data.get("model", {})
    dataset_data = data.get("dataset", {})
    prep_data = data.get("preprocessing", {})
    train_data = data.get("training", {})
    api_data = data.get("api", {})
    voice_data = data.get("voice", {})
    
    return AppConfig(
        model=ModelConfig(**model_data),
        dataset=DatasetConfig(**dataset_data),
        preprocessing=PreprocessingConfig(**prep_data),
        training=TrainingConfig(**train_data),
        api=ApiConfig(**api_data),
        voice=VoiceConfig(**voice_data)
    )
