import os
import yaml
import logging
from dataclasses import dataclass, field, fields
from typing import Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 3
    max_length: int = 128
    dropout: float = 0.1
    device: str = field(default_factory=lambda: "cuda" if os.environ.get("DEVICE") is None and hasattr(os, "environ") else os.environ.get("DEVICE", "cpu"))

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
    api_url: str = "http://127.0.0.1:8000"

@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)

def filter_dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """Filters a dictionary to keep only keys that are fields of the dataclass."""
    cls_fields = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in cls_fields}

def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    """Loads configuration from YAML file, merges environment variables, and returns AppConfig."""
    # Find config relative to project root if needed
    if not os.path.isabs(config_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_path = os.path.join(current_dir, config_path)
        if os.path.exists(possible_path):
            config_path = possible_path
        elif os.path.exists(config_path):
            pass
        else:
            config_path = os.path.join(os.getcwd(), config_path)

    data = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load yaml config at {config_path}: {e}. Using defaults.")
    
    model_data = data.get("model", {})
    dataset_data = data.get("dataset", {})
    prep_data = data.get("preprocessing", {})
    train_data = data.get("training", {})
    api_data = data.get("api", {})
    voice_data = data.get("voice", {})

    # Apply environment variable overrides
    # Port override
    env_port = os.environ.get("PORT") or os.environ.get("API_PORT")
    if env_port:
        try:
            api_data["port"] = int(env_port)
        except ValueError:
            logger.warning(f"Invalid PORT env variable: {env_port}. Ignoring.")
            
    # Host override
    env_host = os.environ.get("HOST") or os.environ.get("API_HOST")
    if env_host:
        api_data["host"] = env_host

    # Device override
    env_device = os.environ.get("DEVICE")
    if env_device:
        model_data["device"] = env_device

    # Model path override
    env_model_path = os.environ.get("MODEL_PATH")
    if env_model_path:
        train_data["output_dir"] = os.path.dirname(env_model_path)
        # We can also store this on a custom field if needed, but output_dir holds the base path

    return AppConfig(
        model=ModelConfig(**filter_dataclass_kwargs(ModelConfig, model_data)),
        dataset=DatasetConfig(**filter_dataclass_kwargs(DatasetConfig, dataset_data)),
        preprocessing=PreprocessingConfig(**filter_dataclass_kwargs(PreprocessingConfig, prep_data)),
        training=TrainingConfig(**filter_dataclass_kwargs(TrainingConfig, train_data)),
        api=ApiConfig(**filter_dataclass_kwargs(ApiConfig, api_data)),
        voice=VoiceConfig(**filter_dataclass_kwargs(VoiceConfig, voice_data))
    )
