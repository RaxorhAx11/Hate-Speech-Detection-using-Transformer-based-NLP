import os
import logging
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from preprocessing import preprocess_text
from config import load_config, AppConfig

logger = logging.getLogger(__name__)

# Label Map
LABEL_MAP = {0: "Safe", 1: "Offensive", 2: "Hate Speech"}

class HateSpeechInference:
    def __init__(self, model_path: str = None):
        config = load_config()
        self.max_length = config.model.max_length
        self.model_path = model_path or os.path.join(config.training.output_dir, "best_model")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path {self.model_path} not found. Train the model first.")
            
        logger.info(f"Loading tokenizer and model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # Load model with output_attentions=True to support interpretability
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            output_attentions=True
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Runs inference on raw text.
        Returns:
            Dictionary containing prediction class, confidence, probabilities, and token importance.
        """
        # Preprocess text
        clean_text = preprocess_text(text, language_filter=None)
        if not clean_text.strip():
            # If text is empty after preprocessing (e.g. only URLs/mentions, or non-English), fallback
            clean_text = text.strip() or "empty input"

        # Tokenize text
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        # Keep input ids for token conversion
        input_ids = inputs["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            attentions = outputs.attentions  # Tuple of layers: (batch, heads, seq, seq)
            
            # Compute probabilities
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
        pred_class_id = int(np.argmax(probs))
        prediction = LABEL_MAP[pred_class_id]
        confidence = float(probs[pred_class_id] * 100)
        
        probabilities_dict = {
            LABEL_MAP[i]: float(probs[i] * 100) for i in range(3)
        }
        
        # Compute Token Importance via Self-Attention
        token_importance = self._compute_attention_importance(tokens, attentions)
        
        # Try SHAP explanation if package is available
        shap_explanation = None
        try:
            shap_explanation = self._explain_with_shap(clean_text)
        except Exception as e:
            logger.debug(f"SHAP explanation skipped or failed: {e}")
            
        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {k: round(v, 2) for k, v in probabilities_dict.items()},
            "tokens": tokens,
            "token_importance": token_importance
        }
        
        if shap_explanation:
            result["shap_importance"] = shap_explanation
            
        return result

    def _compute_attention_importance(self, tokens: List[str], attentions: Tuple[torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Computes the importance of each token by aggregating attention weights across all layers and heads.
        Specifically, we look at the attention received from the CLS token or average attention.
        """
        if not attentions:
            return []
            
        # shape of each layer attention: (1, num_heads, seq_len, seq_len)
        # Average attention weights across all layers and all heads
        layers_att = [layer.cpu().numpy()[0] for layer in attentions] # List of (heads, seq, seq)
        mean_att = np.mean(np.array(layers_att), axis=(0, 1)) # Shape: (seq_len, seq_len)
        
        # CLS token is at index 0. The attention the CLS token pays to other tokens is a standard
        # proxy for their contribution to the classification decision.
        cls_attention = mean_att[0] # Attention from CLS to all tokens
        
        importance_scores = []
        for i, token in enumerate(tokens):
            # Skip special tokens in display importance
            is_special = token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]
            score = float(cls_attention[i])
            importance_scores.append({
                "token": token,
                "importance_score": round(score, 4),
                "is_special": is_special
            })
            
        return importance_scores

    def _explain_with_shap(self, clean_text: str) -> List[Dict[str, Any]]:
        """Generates SHAP values for the input text if SHAP is installed."""
        try:
            import shap
            
            # Custom prediction function for SHAP
            def predict_proba(texts):
                inputs = self.tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                return probs

            # We use PartitionExplainer which is fast and supports text
            explainer = shap.Explainer(predict_proba, self.tokenizer)
            shap_values = explainer([clean_text])
            
            # Extract values for the predicted class
            predicted_class_idx = np.argmax(predict_proba([clean_text])[0])
            values = shap_values.values[0][:, predicted_class_idx]
            data_tokens = shap_values.data[0]
            
            importance = []
            for token, val in zip(data_tokens, values):
                importance.append({
                    "token": token,
                    "shap_value": float(val)
                })
            return importance
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"SHAP explainer encountered error: {e}")
            return None

if __name__ == "__main__":
    # Test inference if best model exists
    try:
        engine = HateSpeechInference()
        res = engine.predict("I hate you so much, you are the worst!")
        print("Prediction result:")
        print(res)
    except Exception as e:
        print(f"Inference testing skipped: {e}")
