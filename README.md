# Advanced Hate Speech Detection using Transformer-based NLP

This repository contains a production-grade NLP pipeline designed to classify text comments into three categories:
1. **Safe** (non-offensive, clean comments)
2. **Offensive** (profane, rude, or toxic comments, but not targeted identity hate)
3. **Hate Speech** (targeted attacks against protected groups, identity-based insults, or threats of violence)

The project leverages modern transformer architectures (defaulting to `distilbert-base-uncased` for resource efficiency on CPU, but fully supporting `roberta-base`, `microsoft/deberta-v3-large`, and `roberta-large` via config) to maximize accuracy and Macro F1 score on imbalanced web text.

---

## Project Directory Structure

The project has been laid out according to professional conventions:

```
project/
│
├── configs/
│   └── config.yaml           # Hyperparameters, model target, and preprocessing configurations
│
├── dataset/                  # Contains raw and final processed dataset splits
│
├── evaluation_plots/         # Confusion matrix, classification report, ROC/PR curve plots
│
├── saved_models/             # Checkpoints and best model weights saved during training
│
├── config.py                 # Python dataclass loading configurations from configs/config.yaml
├── preprocessing.py          # Advanced cleaning pipeline (HTML, URL, emoji, camel case hashtag split)
├── dataset_builder.py        # Compiles, dedups, and maps the 5 source datasets
├── train.py                  # PyTorch training script with custom class weights and tuning
├── evaluate.py               # Generates test set metrics, plots, and misclassified analysis
├── inference.py              # Single/batch prediction wrapper with Attention and SHAP explanations
├── app.py                    # Production FastAPI server exposing /predict endpoint
├── voice.py                  # Voice control center with Speech-to-Text and Text-to-Speech
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Dataset Merging & Label Space Mapping

We unify annotations from **five** distinct public hate speech and toxicity datasets into a single target label space:

| Source Dataset | Original Class | Unified Category | Description |
| :--- | :--- | :--- | :--- |
| **Davidson et al.** | `2 (neither)` / `1 (offensive)` / `0 (hate)` | **Safe** (0) / **Offensive** (1) / **Hate Speech** (2) | Standard mapping. |
| **OLID** (Subtask A) | `NOT` / `OFF` | **Safe** (0) / **Offensive** (1) | No explicit hate category. |
| **HateXplain** | `1 (normal)` / `2 (offensive)` / `0 (hatespeech)` | **Safe** (0) / **Offensive** (1) / **Hate Speech** (2) | Majority vote mapping. |
| **Jigsaw** | multi-label flags | **Safe** (0) / **Offensive** (1) / **Hate Speech** (2) | Identity attack / threats map to Hate Speech. |
| **Civil Comments** | toxicity rates (0 to 1) | **Safe** (0) / **Offensive** (1) / **Hate Speech** (2) | 0.5 threshold mapping. |

---

## Pipeline Execution Workflow

Follow these steps to run the pipeline end-to-end:

### 1. Installation

Install all required Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Building (Modular Prep Pipeline)

Execute the end-to-end dataset preparation pipeline using the orchestrator:
```bash
python scripts/run_pipeline.py
```
This single command runs all pipeline steps sequentially. Alternatively, you can run the individual modular scripts in the scripts folder:
* **Download Raw Data**: `python scripts/download_datasets.py` (Downloads/caches raw Davidson, OLID, HateXplain, Jigsaw, and Civil Comments datasets under `dataset/raw/`)
* **Validate Structure**: `python scripts/validate_dataset.py` (Performs columns, encoding, and corruption validation checks, exporting `dataset/reports/validation_report.md`)
* **Normalize Labels**: `python scripts/normalize_labels.py` (Standardizes all label schemas into 3 classes: Safe (0), Offensive (1), and Hate Speech (2), exporting `dataset/processed/label_mapping.json`)
* **Clean Text**: `python scripts/clean_dataset.py` (Applies optimized regex preprocessed cleaning and English-only language checks, generating `dataset/reports/cleaning_report.md`)
* **Merge & Deduplicate**: `python scripts/merge_datasets.py` (Orchestrates exact/near-duplicate Jaccard removal and resolves label conflicts using majority vote or severity priority rules, exporting `dataset/reports/conflict_report.md` and saving the final unified dataset)
* **Split Stratified**: `python scripts/split_dataset.py` (Generates stratified, leakage-free `train.csv`, `validation.csv`, and `test.csv` splits inside the `dataset/` folder)
* **Generate Reports**: `python scripts/generate_reports.py` (Calculates statistics and exports visualizations like class distribution, sentence lengths, top words, and pipeline stage sizes under `dataset/reports/`)
* **Quality Assurance**: `python scripts/verify_qa.py` (Verifies final dataset UTF-8 compliance, label validity, zero nulls, and zero duplicate overlap/leakage between train, validation, and test splits)

### 3. Model Training

Train the model and optimize hyperparameters:
```bash
python train.py
```
This script:
* Detects GPU acceleration (Mixed Precision FP16 enabled automatically if CUDA is available).
* Performs a lightweight hyperparameter grid/random search (tuning Learning Rate, Batch Size, and Weight Decay) on a data subset to find the configuration that maximizes Macro F1.
* Calculates training class weights to balance the loss function.
* Trains the full dataset with the best parameters, utilizing Hugging Face Trainer and early stopping.
* Saves the best model weights and tokenizer to `saved_models/best_model`.

### 4. Model Evaluation

Generate evaluation plots and analysis metrics:
```bash
python evaluate.py
```
This loads the best model and computes test set accuracy, precision, recall, and Macro F1. It saves:
* `evaluation_plots/classification_report.txt`
* `evaluation_plots/confusion_matrix.png`
* `evaluation_plots/roc_curve.png`
* `evaluation_plots/precision_recall_curve.png`
* `evaluation_plots/misclassified_analysis.csv` (contains a confidence-sorted list of predictions where the model erred).

### 5. Production API Deployment

Run the FastAPI backend server:
```bash
python app.py
```
By default, the server runs on `http://127.0.0.1:8000`. You can configure host, port, device, or model path using environment variables:
| Environment Variable | Description | Example |
| :--- | :--- | :--- |
| `PORT` or `API_PORT` | Port to run the server on (default: `8000`) | `PORT=8080` |
| `HOST` or `API_HOST` | Host address to bind the server to (default: `127.0.0.1`) | `HOST=0.0.0.0` |
| `DEVICE` | Force device selection: `cpu` or `cuda` (default: auto-detects CUDA) | `DEVICE=cpu` |
| `MODEL_PATH` | Path to the best trained transformer directory | `MODEL_PATH=saved_models/best_model` |

---

## Production REST API Reference

The server exposes 4 production endpoints:

### 1. GET `/health`
Verifies that the API service is running, details if the model is currently loaded in memory, and states the inference hardware device.
- **Request Example**:
  ```bash
  curl -s http://127.0.0.1:8000/health
  ```
- **Response Format**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu",
    "model_name": "distilbert-base-uncased",
    "timestamp": 1785987519.28
  }
  ```

### 2. GET `/model-info`
Exposes the model configuration metadata, architecture settings, and supported output labels.
- **Request Example**:
  ```bash
  curl -s http://127.0.0.1:8000/model-info
  ```
- **Response Format**:
  ```json
  {
    "model_name": "distilbert-base-uncased",
    "num_labels": 3,
    "max_length": 128,
    "dropout": 0.1,
    "labels": ["Safe", "Offensive", "Hate Speech"],
    "device": "cpu",
    "model_path": "D:\\project\\saved_models\\best_model",
    "model_loaded": true
  }
  ```

### 3. POST `/predict`
Analyzes a single input text and classifies it. Requests are validated via Pydantic to ensure text is present and non-empty.
- **Request Format**:
  ```json
  {
    "text": "I hate you so much!"
  }
  ```
- **Invocations**:
  - **Bash / Curl**:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"text": "I hate you so much!"}' http://127.0.0.1:8000/predict
    ```
  - **PowerShell**:
    ```powershell
    Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"text": "I hate you so much!"}'
    ```
- **Response Format**:
  ```json
  {
    "prediction": "Offensive",
    "confidence": 49.01,
    "probabilities": {
      "Safe": 3.55,
      "Offensive": 49.01,
      "Hate Speech": 47.43
    },
    "processing_time_ms": 32.58
  }
  ```

### 4. POST `/batch-predict`
Runs optimized batch inference on a list of texts in a single forward pass, returning prediction metrics and aggregated performance timing.
- **Request Format**:
  ```json
  {
    "texts": [
      "I love programming.",
      "Get out of here you jerk!"
    ]
  }
  ```
- **Invocations**:
  - **Bash / Curl**:
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"texts": ["I love programming.", "Get out of here you jerk!"]}' http://127.0.0.1:8000/batch-predict
    ```
  - **PowerShell**:
    ```powershell
    Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/batch-predict -ContentType "application/json" -Body '{"texts": ["I love programming.", "Get out of here you jerk!"]}' | ConvertTo-Json -Depth 5
    ```
- **Response Format**:
  ```json
  {
    "predictions": [
      {
        "prediction": "Safe",
        "confidence": 99.05,
        "probabilities": {
          "Safe": 99.05,
          "Offensive": 0.72,
          "Hate Speech": 0.23
        },
        "processing_time_ms": 17.99
      },
      {
        "prediction": "Offensive",
        "confidence": 90.97,
        "probabilities": {
          "Safe": 0.37,
          "Offensive": 90.97,
          "Hate Speech": 8.65
        },
        "processing_time_ms": 17.99
      }
    ],
    "total_processing_time_ms": 35.97
  }
  ```

---

### 6. Voice and Text Command Center

Execute the voice console client:
```bash
python voice.py
```
This client offers four interactive menu options:
1. **Text input prediction**: Prompts for keyboard text and queries `/predict`.
2. **Voice input prediction (Single phrase)**: Records from your default microphone (with sounddevice fallback), performs Google Speech-to-Text translation, queries the backend API, and reads the classification result out loud using offline Text-to-Speech (TTS).
3. **Continuous voice recognition loop**: Runs a continuous microphone audio stream, automatically adjusting for noise once, analyzing text on-the-fly via the API, and speaking predictions back to you. Say `"stop continuous"`, `"exit"`, or `"quit"` to end the loop.
4. **Exit**: Closes the application.

*If the API server is down, `voice.py` automatically falls back to loading model checkpoints locally to maintain offline support.*

---

### 7. Web Frontend Application (React + TypeScript)

We have added a modern, minimalist React + Vite + TypeScript + Tailwind CSS v4 frontend dashboard located in the `frontend/` directory.

#### Running the Full Stack in Dev Mode

To run both services in parallel:

1. **Start the Backend API Server**:
   From the repository root directory:
   ```bash
   python app.py
   ```
   The backend server runs on `http://127.0.0.1:8000`.

2. **Start the React Frontend Dev Server**:
   Open another terminal, navigate to the `frontend/` directory:
   ```bash
   npm run dev
   ```
   The dev server runs on `http://localhost:5173/`.

Open `http://localhost:5173/` in your web browser to access the visual workspace.

#### Compiling for Production
To compile and bundle the React code into optimized, static HTML/JS/CSS assets:
```bash
npm run build
```
Static production files will be output to the `frontend/dist/` directory.

---

---

## Interpretability Feature

During single inference (`inference.py`), the model aggregates attention weights from self-attention layers to trace which tokens contributed most to the prediction. If you instantiate the engine with `explain=True` (and the `shap` package is available), it will return SHAP attribution scores per token.

---

## Running Automated Tests

Run the complete backend test suite using python's unittest runner:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
This verifies model configuration loading, endpoint health, request parameter validation, batch tokenizations, error handlers, and client voice failovers. All tests execute deterministically on CPU.
