import os
import time
import logging
import json
from typing import Dict, Any, List
import gradio as gr
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from fastapi.middleware.cors import CORSMiddleware
from inference import HateSpeechInference, LABEL_MAP
from config import load_config

# Load config
config = load_config()

# Setup structured logging
log_dir = config.training.logging_dir or "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "api.log")

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)
logger = logging.getLogger("api_logger")

# Initialise FastAPI App
app = FastAPI(
    title="Advanced Hate Speech Detection API",
    description="Production-grade API to classify text into Safe, Offensive, or Hate Speech using transformer models.",
    version="1.0.0"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev/local simplicity. Can be hardened if needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inference engine instance placeholder
inference_engine = None

# Define API schemas
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The raw input text to analyze for hate speech", json_schema_extra={"example": "I hate you so much!"})

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of raw input texts to analyze", json_schema_extra={"example": ["I hate you!", "I love programming."]})

class Probabilities(BaseModel):
    Safe: float = Field(..., description="Probability of safe text in percentage")
    Offensive: float = Field(..., description="Probability of offensive text in percentage")
    Hate_Speech: float = Field(..., alias="Hate Speech", description="Probability of hate speech in percentage")

    model_config = ConfigDict(populate_by_name=True)

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="The predicted class label (Safe, Offensive, or Hate Speech)")
    confidence: float = Field(..., description="Confidence score of the prediction in percentage")
    probabilities: Probabilities = Field(..., description="Detailed class probability distribution")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")

# Helper to resolve model path from environment or local directory
def get_model_target() -> str:
    """Returns local path or Hugging Face Hub ID for the model."""
    return os.environ.get("HF_MODEL_ID") or os.environ.get("MODEL_PATH") or os.path.join(config.training.output_dir, "best_model")

# Startup model loading
@app.on_event("startup")
def load_model_on_startup():
    """Loads the model and tokenizer into memory on API startup."""
    global inference_engine
    model_target = get_model_target()
    
    is_local = os.path.exists(model_target)
    is_hf_hub = "/" in model_target and not os.path.isabs(model_target)
    
    if not is_local and not is_hf_hub:
        logger.warning(
            json.dumps({
                "event": "model_load_skipped",
                "reason": "checkpoint_missing",
                "path": model_target,
                "msg": "Model target not found. Predictions will fail until model is provided."
            })
        )
    else:
        try:
            inference_engine = HateSpeechInference(model_path=model_target)
            logger.info(
                json.dumps({
                    "event": "model_loaded",
                    "path": model_target,
                    "device": str(inference_engine.device)
                })
            )
        except Exception as e:
            logger.error(
                json.dumps({
                    "event": "model_load_failed",
                    "error": str(e)
                })
            )

# Custom Request Validation Error Handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_details = exc.errors()
    logger.warning(
        json.dumps({
            "event": "validation_error",
            "path": request.url.path,
            "errors": error_details
        })
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": [{"field": e["loc"][-1], "message": e["msg"]} for e in error_details]
        }
    )

# Custom General Exception Handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(
        json.dumps({
            "event": "unhandled_exception",
            "path": request.url.path,
            "error": str(exc)
        }, default=str),
        exc_info=True
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Request logger middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000
    
    # Exclude health checks from verbose logging to keep logs clean
    if request.url.path != "/health":
        logger.info(
            json.dumps({
                "event": "api_request",
                "client_ip": request.client.host if request.client else "unknown",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2)
            })
        )
    return response

# GET /health
@app.get("/health", summary="Health Check")
def health_check():
    """Verifies that the API service is running and model is loaded."""
    model_loaded = inference_engine is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": str(inference_engine.device) if model_loaded else None,
        "model_name": config.model.model_name,
        "timestamp": time.time()
    }

# GET /model-info
@app.get("/model-info", summary="Model Metadata Information")
def model_info():
    """Returns metadata details about the configured and loaded transformer model."""
    model_loaded = inference_engine is not None
    model_path = os.path.join(config.training.output_dir, "best_model")
    
    return {
        "model_name": config.model.model_name,
        "num_labels": config.model.num_labels,
        "max_length": config.model.max_length,
        "dropout": config.model.dropout,
        "labels": list(LABEL_MAP.values()),
        "device": str(inference_engine.device) if model_loaded else "not loaded",
        "model_path": os.path.abspath(model_path) if os.path.exists(model_path) else "not found",
        "model_loaded": model_loaded
    }

# GET /metrics
@app.get("/metrics", summary="Model Evaluation Metrics")
def evaluation_metrics():
    """Returns the evaluation metrics from classification_report.txt."""
    import re
    eval_dir = "evaluation_plots"
    report_path = os.path.join(eval_dir, "classification_report.txt")
    
    if not os.path.exists(report_path):
        return {
            "available": False,
            "error": "Classification report not found. Run evaluate.py first."
        }
        
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Parse metrics using regex
        accuracy_match = re.search(r"Accuracy:\s*([\d\.]+)", content)
        precision_match = re.search(r"Precision \(Macro\):\s*([\d\.]+)", content)
        recall_match = re.search(r"Recall \(Macro\):\s*([\d\.]+)", content)
        f1_match = re.search(r"F1 \(Macro\):\s*([\d\.]+)", content)
        roc_auc_match = re.search(r"ROC-AUC \(Macro\):\s*([\d\.]+)", content)
        
        # Parse per-class metrics
        class_metrics = {}
        for label in ["Safe", "Offensive", "Hate Speech"]:
            pattern = rf"{label}\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)"
            class_match = re.search(pattern, content)
            if class_match:
                class_metrics[label] = {
                    "precision": float(class_match.group(1)),
                    "recall": float(class_match.group(2)),
                    "f1_score": float(class_match.group(3)),
                    "support": int(class_match.group(4))
                }
                
        return {
            "available": True,
            "accuracy": float(accuracy_match.group(1)) if accuracy_match else 0.0,
            "precision_macro": float(precision_match.group(1)) if precision_match else 0.0,
            "recall_macro": float(recall_match.group(1)) if recall_match else 0.0,
            "f1_macro": float(f1_match.group(1)) if f1_match else 0.0,
            "roc_auc_macro": float(roc_auc_match.group(1)) if roc_auc_match else 0.0,
            "class_metrics": class_metrics
        }
    except Exception as e:
        return {
            "available": False,
            "error": f"Failed to parse classification report: {str(e)}"
        }


# POST /predict
@app.post("/predict", response_model=PredictionResponse, summary="Analyze text for hate speech")
def predict_hate_speech(request: PredictionRequest):
    """
    Analyzes input text and classifies it as Safe, Offensive, or Hate Speech.
    """
    global inference_engine
    if inference_engine is None:
        model_target = get_model_target()
        try:
            inference_engine = HateSpeechInference(model_path=model_target)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model is not available. Please verify model files or HF_MODEL_ID. Error: {e}"
            )
            
    # Perform prediction
    res = inference_engine.predict(request.text)
    
    probs = res["probabilities"]
    response = PredictionResponse(
        prediction=res["prediction"],
        confidence=res["confidence"],
        probabilities=Probabilities(
            Safe=probs.get("Safe", 0.0),
            Offensive=probs.get("Offensive", 0.0),
            **{"Hate Speech": probs.get("Hate Speech", 0.0)}
        ),
        processing_time_ms=res["processing_time_ms"]
    )
    
    # Structured logging for predictions
    logger.info(
        json.dumps({
            "event": "prediction_made",
            "text_length": len(request.text),
            "prediction": res["prediction"],
            "confidence": res["confidence"],
            "processing_time_ms": res["processing_time_ms"]
        })
    )
    return response

# POST /batch-predict
@app.post("/batch-predict", response_model=BatchPredictionResponse, summary="Analyze batch of texts for hate speech")
def batch_predict_hate_speech(request: BatchPredictionRequest):
    """
    Analyzes a list of input texts in an optimized single batch.
    """
    global inference_engine
    if inference_engine is None:
        model_target = get_model_target()
        try:
            inference_engine = HateSpeechInference(model_path=model_target)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model is not available. Please verify model files or HF_MODEL_ID. Error: {e}"
            )
            
    # Run batch prediction
    res = inference_engine.predict_batch(request.texts)
    
    predictions_response = []
    for pred in res["predictions"]:
        probs = pred["probabilities"]
        predictions_response.append(
            PredictionResponse(
                prediction=pred["prediction"],
                confidence=pred["confidence"],
                probabilities=Probabilities(
                    Safe=probs.get("Safe", 0.0),
                    Offensive=probs.get("Offensive", 0.0),
                    **{"Hate Speech": probs.get("Hate Speech", 0.0)}
                ),
                processing_time_ms=pred["processing_time_ms"]
            )
        )
        
    response = BatchPredictionResponse(
        predictions=predictions_response,
        total_processing_time_ms=res["total_processing_time_ms"]
    )
    
    # Structured logging for batch predictions
    logger.info(
        json.dumps({
            "event": "batch_prediction_made",
            "batch_size": len(request.texts),
            "total_processing_time_ms": res["total_processing_time_ms"]
        })
    )
    return response

# ---------------------------------------------------------------------------
# Gradio UI Integration (100% Free on Hugging Face Spaces SDK: gradio)
# ---------------------------------------------------------------------------
def gradio_predict(text: str):
    if not text or not text.strip():
        return "Please enter text to analyze.", "0.0%", {}, "0.0 ms"
    req = PredictionRequest(text=text)
    res = predict_hate_speech(req)
    probs = {
        "Safe": round(res.probabilities.Safe / 100.0, 4),
        "Offensive": round(res.probabilities.Offensive / 100.0, 4),
        "Hate Speech": round(getattr(res.probabilities, "Hate Speech") / 100.0, 4),
    }
    return res.prediction, f"{res.confidence:.1f}%", probs, f"{res.processing_time_ms:.1f} ms"

def gradio_batch_predict(text_block: str):
    lines = [line.strip() for line in (text_block or "").split("\n") if line.strip()]
    if not lines:
        return [["No input lines", "-", "-", "-"]]
    req = BatchPredictionRequest(texts=lines)
    res = batch_predict_hate_speech(req)
    rows = []
    for pred in res.predictions:
        hate_prob = getattr(pred.probabilities, "Hate Speech")
        rows.append([
            pred.prediction,
            f"{pred.confidence:.1f}%",
            f"Safe: {pred.probabilities.Safe:.1f}% | Off: {pred.probabilities.Offensive:.1f}% | Hate: {hate_prob:.1f}%",
            f"{pred.processing_time_ms:.1f} ms"
        ])
    return rows

with gr.Blocks(title="Hate Speech Detection AI") as demo:
    gr.Markdown(
        """
        # 🛡️ Advanced Hate Speech Detection AI
        ### Production Transformer NLP Pipeline (Safe vs. Offensive vs. Hate Speech)
        *Trained on 5 unified toxicity benchmarks: Davidson, OLID, HateXplain, Jigsaw, and Civil Comments.*
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("🔍 Single Text Scanner"):
            with gr.Row():
                with gr.Column(scale=3):
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter comment or message to analyze...",
                        lines=3
                    )
                    with gr.Row():
                        scan_btn = gr.Button("🚀 Analyze Text", variant="primary")
                        clear_btn = gr.Button("Clear")
                    
                    gr.Examples(
                        examples=[
                            ["Have a wonderful and blessed day!"],
                            ["This movie was so terrible and boring."],
                            ["You are an absolute idiot, shut up!"],
                            ["Go back to your country, you disgusting animals."]
                        ],
                        inputs=text_input,
                        label="Quick Test Examples (Click to test)"
                    )
                with gr.Column(scale=2):
                    pred_output = gr.Textbox(label="Predicted Category", interactive=False)
                    conf_output = gr.Textbox(label="Confidence", interactive=False)
                    probs_output = gr.Label(label="Class Probability Distribution", num_top_classes=3)
                    time_output = gr.Textbox(label="Latency", interactive=False)
                    
            scan_btn.click(
                fn=gradio_predict,
                inputs=[text_input],
                outputs=[pred_output, conf_output, probs_output, time_output]
            )
            clear_btn.click(lambda: ("", "", {}, ""), outputs=[text_input, pred_output, probs_output, time_output])

        with gr.TabItem("📑 Batch Analyzer"):
            batch_input = gr.Textbox(
                label="Enter multiple sentences (one per line)",
                placeholder="First comment\nSecond comment\nThird comment...",
                lines=5
            )
            batch_btn = gr.Button("⚡ Analyze Batch", variant="primary")
            batch_output = gr.Dataframe(
                headers=["Prediction", "Confidence", "Probability Breakdown", "Latency"],
                datatype=["str", "str", "str", "str"],
                label="Batch Prediction Results"
            )
            batch_btn.click(fn=gradio_batch_predict, inputs=[batch_input], outputs=[batch_output])
            
        with gr.TabItem("📊 Model Metrics & Trust"):
            gr.Markdown(
                """
                ### Stratified Holdout Test Set Performance (1,500 samples)
                - **Accuracy**: 85.73%
                - **Macro F1 Score**: 85.54%
                - **Macro Precision**: 85.91%
                - **Macro Recall**: 85.73%
                - **Macro ROC-AUC**: 95.74%
                """
            )
            cm_path = os.path.join("evaluation_plots", "confusion_matrix.png")
            if os.path.exists(cm_path):
                gr.Image(value=cm_path, label="Evaluation Confusion Matrix")

        with gr.TabItem("🌐 API Documentation"):
            gr.Markdown(
                """
                ### REST API Endpoints
                This server also exposes production REST API endpoints:
                - `POST /predict` - Single text inference
                - `POST /batch-predict` - Batch texts inference
                - `GET /health` - Service & model health check
                - `GET /metrics` - Real-time model evaluation metrics
                - **Interactive Swagger Documentation**: Access [`/docs`](/docs) on this server.
                """
            )

# Mount Gradio onto the existing FastAPI application at root /
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    # Use API configuration settings (with env variable fallback applied in config.py)
    uvicorn.run("app:app", host=config.api.host, port=config.api.port, reload=False)
