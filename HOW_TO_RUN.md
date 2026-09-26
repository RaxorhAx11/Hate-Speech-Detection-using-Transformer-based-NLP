# How to Run the Hate Speech Detection Project

This guide provides step-by-step instructions to run the entire pipeline, train models, deploy the FastAPI backend, run the React frontend, and run the voice assistant interface.

---

## 1. Installation & Setup

Ensure you have Python 3.8+ and Node.js (v16+) installed.

### Python Backend Dependencies
From the root directory, install all required Python libraries:
```bash
pip install -r requirements.txt
```

### React Frontend Dependencies
Navigate to the `frontend/` directory and install the Node modules:
```bash
cd frontend
npm install
cd ..
```

---

## 2. Dataset Preparation (ETL Pipeline)
Compile and build the unified dataset from the five source datasets (OLID, Davidson, HateXplain, Jigsaw, Civil Comments):

To run the entire modular pipeline sequentially with a single command:
```bash
python scripts/run_pipeline.py
```
*This handles downloading raw data, normalizing labels, cleaning text, deduplicating, splitting splits (Train/Val/Test), and generating validation QA reports.*

---

## 3. Model Training & Evaluation

### Train the Transformer Model
To perform hyperparameter tuning and train the final classifier:
```bash
python train.py
```
*The script automatically selects GPU (CUDA) if available. The best model checkpoints will be saved in `saved_models/best_model`.*

### Run Model Evaluation
Compute test metrics (Accuracy, Precision, Recall, Macro F1) and generate performance curves:
```bash
python evaluate.py
```
*This saves confusion matrices and precision-recall charts inside `evaluation_plots/` and updates `assets/confusion_matrix.png`.*

---

## 4. Launching the Backend REST API
Start the FastAPI server:
```bash
python app.py
```
* The API runs locally on **`http://127.0.0.1:8000`**.
* The server will automatically load the model saved in `saved_models/best_model` on startup.

---

## 5. Launching the React Web Frontend
Open a new terminal window, navigate to the `frontend/` directory, and start the Vite dev server:
```bash
cd frontend
npm run dev
```
* Open your browser and go to **`http://localhost:5173/`** to interact with the prediction sandbox, upload files for batch processing, view model evaluation curves, or view inference history.

---

## 6. Voice and Text Command Interface
Start the interactive CLI client:
```bash
python voice.py
```
The console client provides:
1. **Text input prediction**: Query predictions via standard keyboard typing.
2. **Voice input prediction**: Speak into your microphone to get prediction labels read back to you.
3. **Continuous voice loop**: Stream voice inputs continuously. (To exit this mode, say *"stop continuous"*, *"exit"*, or *"quit"*).

---

## 7. Running Backend Unit Tests
To execute backend verification checks:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 8. Free Public Demo Deployment (Hugging Face Spaces)
To deploy this full-stack application online for free for recruiters and portfolio viewing:
1. Review the step-by-step instructions in [DEPLOYMENT.md](DEPLOYMENT.md).
2. Create a free Docker Space on [Hugging Face Spaces](https://huggingface.co/new-space).
3. Push or upload your code to the Space.
4. Your live app will be accessible at:
   `https://<username>-<space-name>.hf.space`

