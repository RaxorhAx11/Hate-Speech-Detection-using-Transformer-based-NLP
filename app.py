import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from inference import HateSpeechInference, LABEL_MAP
from config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load config
config = load_config()

# Define API request schema
class PredictionRequest(BaseModel):
    text: str = Field(..., description="The raw input text to analyze for hate speech", example="I hate you so much!")

# Define API response schema
class Probabilities(BaseModel):
    Safe: float = Field(..., description="Probability of safe text in percentage")
    Offensive: float = Field(..., description="Probability of offensive text in percentage")
    Hate_Speech: float = Field(..., alias="Hate Speech", description="Probability of hate speech in percentage")

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="The predicted class label (Safe, Offensive, or Hate Speech)")
    confidence: float = Field(..., description="Confidence score of the prediction in percentage")
    probabilities: Probabilities = Field(..., description="Detailed class probability distribution")

# Initialize FastAPI App
app = FastAPI(
    title="Advanced Hate Speech Detection API",
    description="Production-grade API to classify text into Safe, Offensive, or Hate Speech using transformer models.",
    version="1.0.0"
)

# Inference engine instance placeholder
inference_engine = None

@app.on_event("startup")
def load_model_on_startup():
    """Loads the model and tokenizer into memory on API startup."""
    global inference_engine
    model_path = os.path.join(config.training.output_dir, "best_model")
    
    if not os.path.exists(model_path):
        logger.warning(
            f"Best model not found at '{model_path}'. Predictions will fail until the model is trained. "
            "Please run 'python train.py' to train and save the model."
        )
    else:
        try:
            inference_engine = HateSpeechInference(model_path=model_path)
            logger.info("Best model loaded successfully on startup.")
        except Exception as e:
            logger.error(f"Failed to load the model on startup: {e}")

@app.get("/health", summary="Health Check")
def health_check():
    """Verifies that the API service is running and checking if the model is loaded."""
    model_loaded = inference_engine is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "message": "API running. Model is loaded." if model_loaded else "API running. Model is NOT loaded. Run training first."
    }

@app.post("/predict", response_model=PredictionResponse, summary="Analyze text for hate speech")
def predict_hate_speech(request: PredictionRequest):
    """
    Analyzes input text and classifies it as either:
    - **Safe**: Clean, non-offensive text.
    - **Offensive**: Toxic or profane words without targeted hate.
    - **Hate Speech**: Targeted slurs or offensive statements attacking individuals or groups based on identity features.
    """
    global inference_engine
    if inference_engine is None:
        # Try to load on the fly if it wasn't loaded (e.g. if trained after startup)
        model_path = os.path.join(config.training.output_dir, "best_model")
        if os.path.exists(model_path):
            try:
                inference_engine = HateSpeechInference(model_path=model_path)
                logger.info("Model loaded dynamically on request.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
        else:
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please run training (train.py) before attempting predictions."
            )
            
    try:
        result = inference_engine.predict(request.text)
        
        # Prepare response mapping exactly to requirements
        # Note the alias mappings for 'Hate Speech' in Probabilities pydantic class
        probs = result["probabilities"]
        
        response = PredictionResponse(
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=Probabilities(
                Safe=probs.get("Safe", 0.0),
                Offensive=probs.get("Offensive", 0.0),
                **{"Hate Speech": probs.get("Hate Speech", 0.0)}
            )
        )
        return response
    except Exception as e:
        logger.error(f"Inference error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.api.host, port=config.api.port)
