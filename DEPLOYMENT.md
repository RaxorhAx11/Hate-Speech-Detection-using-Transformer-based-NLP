# 100% Free Public Demo Deployment Guide

> [!IMPORTANT]
> **Why Docker is Paid on Hugging Face & Why Gradio SDK is the 100% Free Solution:**
> Hugging Face recently moved the **Docker SDK** behind their paid PRO tier ($9/mo). However, the **Gradio SDK** remains **100% FREE with 16 GB RAM, 2 vCPUs, and NO credit card required**.
> 
> We have updated `app.py` so that when deployed as a **Gradio Space**:
> 1. It automatically launches an interactive web application (Tabs for Single Text Scan, Preset Click Examples, Probability Progress Bars, Batch Analysis, Confusion Matrix, and Metrics).
> 2. It **simultaneously serves the FastAPI REST API** (`/predict`, `/batch-predict`, `/health`, `/metrics`, `/docs`), allowing external frontends (such as your React app on Vercel) to call it!

---

## 1. Free Deployment Architecture Overview

```
                                  [ Recruiters / Browser ]
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [ Hugging Face Space (Gradio) ]                    [ Vercel (React Frontend) ]
         • 100% Free (16 GB RAM, 2 vCPU)                   • 100% Free Edge Hosting
         • URL: https://<user>-<space>.hf.space            • URL: https://<user>-app.vercel.app
         • Interactive UI + FastAPI Endpoints              • Full Custom React UI
                      │                                               │
                      └────────────────── API Calls ──────────────────┘
```

You can use **either or both**:
- **Option 1 (Fastest & Simplest)**: Deploy to **Hugging Face Space (Gradio SDK)**. Recruiters open the Hugging Face URL and can immediately test comments, inspect class probabilities, and view evaluation metrics.
- **Option 2 (Full Portfolio Showcase)**: Host the ML Backend on the **Hugging Face Gradio Space**, and deploy the **React Frontend on Vercel** (which connects to the Space API via `VITE_API_URL`). Both services are 100% free forever.

---

## 2. Option 1: Deploy to Hugging Face Spaces (Gradio SDK - 100% Free)

### Step 1: Create the Space on Hugging Face
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Configure your Space:
   - **Space Name**: `hate-speech-detection`
   - **License**: `mit`
   - **Select the Space SDK**: Select **Gradio** *(Do NOT select Docker; Gradio is 100% Free)*
   - **Space Hardware**: **CPU basic • 2 vCPU • 16 GB • Free**
   - **Visibility**: **Public**
3. Click **Create Space**.

---

### Step 2: Push Your Code to the Space

Open a terminal in your project directory:

```bash
# 1. Enable Git LFS (for model weights)
git lfs install

# 2. Add your Hugging Face Space as a remote
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<YOUR_SPACE_NAME>

# 3. Stage, commit, and push
git add .
git commit -m "Deploy Hate Speech Detection with Gradio and FastAPI"
git push space main
```

*(When prompted for password, enter your Hugging Face Access Token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).*

#### Alternative (Drag-and-Drop via Web UI)
If you prefer not using Git CLI:
1. In your Hugging Face Space, click the **Files** tab -> **Add file** -> **Upload files**.
2. Upload:
   - `app.py`, `config.py`, `inference.py`, `preprocessing.py`
   - `requirements.txt`
   - `README.md`
   - `configs/`
   - `evaluation_plots/`
   - `saved_models/best_model/`
3. Click **Commit changes to main**.

---

### Step 3: Where You Get Your Public URL
Once the Space finishes building (takes ~2 minutes):
1. **Interactive Demo URL**:
   ```
   https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<YOUR_SPACE_NAME>
   ```
2. **Direct Fullscreen URL** (Best for recruiters):
   ```
   https://<YOUR_HF_USERNAME>-<YOUR_SPACE_NAME>.hf.space
   ```
3. **Interactive Swagger API Docs**:
   ```
   https://<YOUR_HF_USERNAME>-<YOUR_SPACE_NAME>.hf.space/docs
   ```

---

## 3. Option 2 (Optional): Deploy the React Frontend to Vercel (100% Free)

If you want recruiters to also see your custom React TypeScript frontend:

1. Push your repository to your GitHub account:
   ```bash
   git push origin main
   ```
2. Go to [vercel.com](https://vercel.com) and click **Add New Project** -> **Import Git Repository**.
3. Select your GitHub repository (`Hate-Speech-Detection-using-Transformer-based-NLP`).
4. In the Project Configuration:
   - **Root Directory**: Click `Edit` and select `frontend`.
   - **Environment Variables**: Add:
     - **Name**: `VITE_API_URL`
     - **Value**: `https://<YOUR_HF_USERNAME>-<YOUR_SPACE_NAME>.hf.space` *(Your Hugging Face Space direct URL)*
5. Click **Deploy**.

Vercel will give you a live production URL (e.g. `https://hate-speech-detection-xxx.vercel.app`) that queries your Hugging Face model in real time.

---

## 4. Local Testing

To run the unified backend with Gradio locally:
```bash
python app.py
```
Open **`http://127.0.0.1:8000`** in your browser.
