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
    sample_size_per_class: int = 5000
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

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
