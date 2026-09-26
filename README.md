---
title: Advanced Hate Speech Detection AI
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

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
├── assets/                   # Project visual assets (confusion matrix plot, etc.)
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
* `assets/confusion_matrix.png` (embedded in README.md)
* `evaluation_plots/classification_report.txt`
* `evaluation_plots/confusion_matrix.png`
* `evaluation_plots/confusion_matrix.csv`
* `evaluation_plots/roc_curve.png`
* `evaluation_plots/precision_recall_curve.png`
* `evaluation_plots/misclassified_analysis.csv` (contains a confidence-sorted list of predictions where the model erred).

---

## Experimental Results & Model Performance

The trained transformer model (`distilbert-base-uncased` fine-tuned with class-weighted cross-entropy loss) was evaluated on the held-out, stratified test set (`dataset/test.csv`, **1,500 samples**: 500 Safe, 500 Offensive, 500 Hate Speech). All metrics were computed directly from model evaluation using `evaluate.py`.

### Overall Summary Metrics

| Metric | Value | Percentage |
| :--- | :---: | :---: |
| **Test Accuracy** | `0.8573` | **85.73%** |
| **Macro F1 Score** | `0.8554` | **85.54%** |
| **Macro Precision** | `0.8591` | **85.91%** |
| **Macro Recall** | `0.8573` | **85.73%** |
| **Weighted F1 Score** | `0.8554` | **85.54%** |
| **Macro ROC-AUC (OVR)** | `0.9574` | **95.74%** |

### Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support (Test Samples) |
| :--- | :---: | :---: | :---: | :---: |
| **Safe (0)** | 0.9018 | 0.9180 | 0.9098 | 500 |
| **Offensive (1)** | 0.8615 | 0.7340 | 0.7927 | 500 |
| **Hate Speech (2)** | 0.8142 | 0.9200 | 0.8638 | 500 |
| **Macro Average** | **0.8591** | **0.8573** | **0.8554** | **1,500** |
| **Weighted Average** | **0.8591** | **0.8573** | **0.8554** | **1,500** |

### Confusion Matrix

The confusion matrix below demonstrates the classification distribution across all 1,500 test set comments:

![Confusion Matrix](assets/confusion_matrix.png)

#### Tabular Confusion Matrix

| True Label \ Predicted Label | Predicted Safe | Predicted Offensive | Predicted Hate Speech | Total True Support |
| :--- | :---: | :---: | :---: | :---: |
| **True Safe** | **459** | 29 | 12 | 500 |
| **True Offensive** | 40 | **367** | 93 | 500 |
| **True Hate Speech** | 10 | 30 | **460** | 500 |
| **Total Predicted** | 509 | 426 | 565 | 1,500 |

#### Key Analytical Observations:
- **High Sensitivity to Hate Speech (92.00% Recall)**: The model identified **460 out of 500** hate speech samples, minimizing harmful false negatives in high-risk moderation scenarios.
- **Accurate Benign Separation (90.18% Precision, 91.80% Recall)**: Safe comments are preserved with minimal false positives; only 12 benign comments were classified as hate speech.
- **Nuanced Boundary Between Offensive & Hate Speech**: The most frequent misclassifications occur between *Offensive* and *Hate Speech* (93 offensive comments predicted as hate speech, 30 hate speech predicted as offensive), reflecting lexical overlap in vulgar language versus identity-targeted abuse.

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

## Bias & Fairness

Automated content moderation systems operate in sensitive domains where classification errors can disproportionately silence marginalized communities or reinforce societal biases. While the fine-tuned transformer achieves strong aggregate classification metrics (85.54% Macro F1), aggregate performance across an entire test set can obscure critical fairness vulnerabilities.

### Why Hate-Speech Classifiers Produce Biased Predictions
Hate speech and toxicity models are inherently susceptible to systematic bias due to two primary factors:
1. **Spurious Lexical Correlations**: Transformer architectures learn statistical associations between frequent token co-occurrences and target labels. If certain group descriptors or vernacular tokens frequently co-occur with toxic comments in the training data, the model learns superficial shortcuts—flagging sentences based on vocabulary rather than true semantic intent, pragmatics, or harassment targets.
2. **Annotator Subjectivity & Cultural Prior**: Public datasets rely on crowdsourced or third-party annotations that reflect the demographic and cultural priors of the raters. Nuanced in-group communication, reclaimable slurs, and regional slang are often misinterpreted and penalized by annotators outside those communities.

### Key Bias & Fairness Risks

#### 1. Racial & Dialectal Bias (AAVE)
Extensive NLP fairness research (e.g., Sap et al., 2019; Blodgett et al., 2020) has shown that hate speech classifiers often display severe racial disparities, consistently predicting higher toxicity probabilities for text containing markers of **African American Vernacular English (AAVE)**. In-group conversational markers, syntactic features (such as habitual *be*), and vernacular expressions are frequently misclassified as offensive or abusive due to annotator bias and skewed training samples.

#### 2. False Positives on Identity-Related Language
A primary failure mode in toxicity moderation is the conflation of **identity-identifying terminology** with hate speech. Mentions of protected groups—including sexual orientation, religion, race, and gender (e.g., *"gay"*, *"lesbian"*, *"Muslim"*, *"Jewish"*, *"Black"*, *"transgender"*)—frequently trigger false-positive hate speech classifications, even in benign, educational, affirming, or counterspeech contexts, because these identity terms disproportionately co-occur with hostility in uncurated web text.

#### 3. Dataset Imbalance & Multi-Source Domain Shift
The unified dataset in this project aggregates five distinct data sources (Davidson et al., OLID, HateXplain, Jigsaw, and Civil Comments). Each corpus differs significantly in its collection strategy (e.g., Twitter keyword sampling vs. Wikipedia Talk page discussions vs. news site comment threads), annotator guidelines, and class prevalence. This heterogeneity introduces sampling bias: overt profanity and slurs are overrepresented compared to subtle, dog-whistle, or structural hate speech, distorting the model's decision boundaries across conversational contexts.

### Subgroup Evaluation Status & Future Roadmap

> [!IMPORTANT]
> **Subgroup fairness and demographic parity have not been comprehensively evaluated on this model.**
> The merged multi-source dataset does not contain standardized, complete demographic or dialectal metadata across all splits. To uphold scientific integrity and avoid false guarantees, no synthetic or unvalidated fairness metrics are presented.

Future iterations of this project will integrate a dedicated fairness evaluation pipeline including:
- **Subgroup Fairness Metrics**: Implementation of specialized fairness metrics introduced in bias benchmarks, such as **Subgroup AUC**, **Background Positive Subgroup Negative (BPSN) AUC**, and **Background Negative Subgroup Positive (BNSP) AUC** across demographic identity slices to measure false-positive parity.
- **Counterfactual & Template Evaluation**: Auditing with synthetic diagnostic suites such as the **Equity Evaluation Corpus (EEC)** and perturbation tests (swapping demographic identity tokens in identical grammatical frames to verify prediction invariance).
- **Dialectal Benchmarking**: Explicitly measuring performance disparities between paired AAVE and Standard American English (SAE) corpora.
- **Targeted Bias Mitigation**: Exploring counterfactual data augmentation (CDA), adversarial debiasing, and demographic-aware threshold calibration prior to any production deployment.

---

## Running Automated Tests

Run the complete backend test suite using python's unittest runner:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
This verifies model configuration loading, endpoint health, request parameter validation, batch tokenizations, error handlers, and client voice failovers. All tests execute deterministically on CPU.

---

## 🚀 Free Live Demo Deployment (Hugging Face Spaces)

This project is configured for **100% free deployment on Hugging Face Spaces** using Docker:
- **Free CPU Hardware**: 2 vCPU, 16 GB RAM, 50 GB storage (generous capacity for PyTorch DistilBERT inference).
- **Single-Port Architecture**: Built-in multi-stage `Dockerfile` compiles the React frontend and serves both the React UI and the FastAPI REST API through port `7860`.
- **Public URL**: Generates a shareable URL (`https://<username>-<space-name>.hf.space`) for recruiters and portfolio showcasing.

For complete step-by-step instructions, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.


