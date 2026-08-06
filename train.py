import os
import random
import logging
import numpy as np
import pandas as pd
import torch
import shutil
from typing import Dict, Any, Tuple, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset
from config import load_config, AppConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WeightedTrainer(Trainer):
    """Custom HuggingFace Trainer that supports class-weighted cross-entropy loss."""
    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
            # Move class weights to the appropriate device
            self.class_weights = self.class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None and labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs[0]
            
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Computes evaluation metrics (accuracy, precision, recall, macro f1, weighted f1)."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate macro and weighted metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        "accuracy": acc,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted
    }

def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Calculates class weights inversely proportional to class frequencies."""
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    num_classes = 3
    # Compute inverse class frequency weights
    weights = []
    for i in range(num_classes):
        class_count = counts.get(i, 0)
        if class_count > 0:
            w = total / (num_classes * class_count)
        else:
            w = 1.0
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float)

def train_model(
    config: AppConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: Dict[str, Any] = None,
    use_weighted_loss: bool = True
) -> Tuple[Trainer, AutoModelForSequenceClassification, Dict[str, float]]:
    """Runs the model training pipeline for a given config and hyperparameters."""
    
    # Extract configs
    model_name = hyperparams.get("model_name", config.model.model_name) if hyperparams else config.model.model_name
    learning_rate = hyperparams.get("learning_rate", config.training.learning_rate) if hyperparams else config.training.learning_rate
    batch_size = hyperparams.get("batch_size", config.training.batch_size) if hyperparams else config.training.batch_size
    weight_decay = hyperparams.get("weight_decay", config.training.weight_decay) if hyperparams else config.training.weight_decay
    dropout = hyperparams.get("dropout", config.model.dropout) if hyperparams else config.model.dropout
    max_length = hyperparams.get("max_length", config.model.max_length) if hyperparams else config.model.max_length
    warmup_ratio = hyperparams.get("warmup_ratio", config.training.warmup_ratio) if hyperparams else config.training.warmup_ratio
    epochs = hyperparams.get("epochs", config.training.epochs) if hyperparams else config.training.epochs

    logger.info(f"Configuring training: Model={model_name}, LR={learning_rate}, BatchSize={batch_size}, Epochs={epochs}")
    
    # Set seed
    set_seed(config.dataset.random_seed)
    
    # Load tokenizer and tokenized dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], truncation=True, max_length=max_length)
    
    # Convert pandas DataFrames to HF datasets
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    
    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds = val_ds.map(tokenize_function, batched=True)
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Load base model config and set dropout dynamically
    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=config.model.num_labels
    )
    
    # Handle different dropout attribute names across model families
    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = dropout
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = dropout
    if hasattr(model_config, "dropout"):
        model_config.dropout = dropout
    if hasattr(model_config, "attention_dropout"):
        model_config.attention_dropout = dropout
        
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=model_config
    )
    
    # Set training arguments
    fp16_enabled = torch.cuda.is_available() and config.training.fp16
    
    # Create unique output directory for run checkpoints
    run_dir = os.path.join(config.training.output_dir, "current_run")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
        
    training_args = TrainingArguments(
        output_dir=run_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        warmup_ratio=warmup_ratio,
        logging_dir=config.training.logging_dir,
        logging_steps=10,
        save_strategy=config.training.save_strategy,
        eval_strategy=config.training.evaluation_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=fp16_enabled,
        report_to="none"
    )
    
    # Class weights for Weighted Loss
    class_weights = None
    if use_weighted_loss:
        class_weights = compute_class_weights(train_df)
        logger.info(f"Computed Class Weights: {class_weights}")
        
    # Instantiate custom or standard trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)]
    )
    
    # Train model
    trainer.train()
    
    # Evaluate model
    eval_metrics = trainer.evaluate()
    logger.info(f"Training completed. Validation Macro F1: {eval_metrics.get('eval_macro_f1'):.4f}")
    
    return trainer, model, eval_metrics

def tune_hyperparameters(config: AppConfig, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """Performs a lightweight random search to find the best hyperparameters."""
    logger.info("Starting hyperparameter tuning...")
    
    # Define hyperparameter search spaces
    lr_space = [1e-5, 2e-5, 3e-5]
    batch_space = [8, 16]
    weight_decay_space = [0.01, 0.1]
    
    best_f1 = -1.0
    best_params = {}
    
    # Use a small subset (1000 samples) of the training data for faster tuning runs
    tuning_subset_size = min(1500, len(train_df))
    tuning_val_size = min(300, len(val_df))
    tuning_train_df = train_df.sample(n=tuning_subset_size, random_state=config.dataset.random_seed)
    tuning_val_df = val_df.sample(n=tuning_val_size, random_state=config.dataset.random_seed)
    
    # Run 3 trial configurations
    num_trials = 3
    for trial in range(num_trials):
        params = {
            "learning_rate": random.choice(lr_space),
            "batch_size": random.choice(batch_space),
            "weight_decay": random.choice(weight_decay_space),
            "epochs": 1,  # Train for only 1 epoch for tuning efficiency
            "model_name": config.model.model_name
        }
        
        logger.info(f"Trial {trial+1}/{num_trials}: {params}")
        try:
            _, _, metrics = train_model(
                config=config,
                train_df=tuning_train_df,
                val_df=tuning_val_df,
                hyperparams=params,
                use_weighted_loss=True
            )
            macro_f1 = metrics.get("eval_macro_f1", 0.0)
            logger.info(f"Trial {trial+1} F1 Score: {macro_f1:.4f}")
            
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_params = params
        except Exception as e:
            logger.error(f"Trial {trial+1} failed: {e}")
            
    logger.info(f"Tuning finished. Best Params: {best_params} with F1: {best_f1:.4f}")
    return best_params

def main():
    config = load_config()
    
    # Load training and validation datasets
    train_path = os.path.join(config.dataset.data_dir, "train.csv")
    val_path = os.path.join(config.dataset.data_dir, "val.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.error(f"Dataset files not found. Please run dataset_builder.py first.")
        return
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Tune hyperparameters
    best_params = tune_hyperparameters(config, train_df, val_df)
    
    # Train full model with best hyperparameters and for full epochs
    best_params["epochs"] = config.training.epochs
    
    logger.info("Training final model with best hyperparameters...")
    trainer, model, metrics = train_model(
        config=config,
        train_df=train_df,
        val_df=val_df,
        hyperparams=best_params,
        use_weighted_loss=True
    )
    
    # Save best model and tokenizer
    best_model_path = os.path.join(config.training.output_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    
    logger.info(f"Saving best model to {best_model_path}")
    model.save_pretrained(best_model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(best_params.get("model_name", config.model.model_name))
    tokenizer.save_pretrained(best_model_path)
    logger.info("Model and tokenizer saved successfully.")

if __name__ == "__main__":
    main()
