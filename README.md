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

### 2. Dataset Building

Execute the dataset compilation script to download, clean, deduplicate, and split the data:
```bash
python dataset_builder.py
```
This downloads Davidson, OLID, Jigsaw, Civil Comments, and HateXplain, applies text preprocessing, resolves conflicting labels, and outputs balanced `train.csv`, `val.csv`, and `test.csv` splits to the `dataset/` directory.

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
The server exposes:
* **GET `/health`**: Health check.
* **POST `/predict`**: Takes JSON input:
  ```json
  { "text": "I hate you so much!" }
  ```
  Returns:
  ```json
  {
    "prediction": "Hate Speech",
    "confidence": 98.42,
    "probabilities": {
      "Safe": 0.3,
      "Offensive": 1.2,
      "Hate Speech": 98.5
    }
  }
  ```

### 6. Voice and Text Command Center

Execute the voice console:
```bash
python voice.py
```
This console allows you to select between text typing and voice recording. When you speak, it converts your voice input to text via Speech-to-Text, classifies the text, and reads the classification out loud via offline Text-to-Speech (TTS).

---

## Interpretability Feature

During inference (`inference.py`), the model aggregates attention weights from self-attention layers to trace which tokens contributed most to the prediction. If the `shap` package is available, it will also return SHAP attribution scores per token.
