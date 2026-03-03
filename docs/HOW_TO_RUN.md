# HOW TO RUN — End-to-End Execution & Deployment Guide

A comprehensive guide to launching the **CSAO (Cart Super Add-On)** project locally and deploying it to the cloud. This walkthrough covers data generation, machine learning training, launching the internal API, and running the interactive **HTML/JS Web Interface**.

---

## Prerequisites

- **Python 3.10+**
- `~1GB` free disk space (data generation requires room for CSVs).
- **Git LFS** installed (required for handling large datasets during pushes).
- **PyTorch** (Optional but recommended) — The GRU component will cleanly bypass itself if PyTorch cannot bind to your hardware.

---

## Step 1: System Initialization & Offline Config

Ensure all dependencies are mapped and installed prior to running the engines:
```bash
# Install core Python packages
pip install -r requirements.txt
```

### Local LLMs vs Cloud APIs

The codebase uses Generative AI in two distinctly different offline processes during dataset creation (`src/llm_components.py`):

1. **Local LLM (Semantic Embeddings):** 
   The system natively downloads and runs a local open-source LLM (`all-MiniLM-L6-v2` via `sentence-transformers`) entirely on your local CPU. It requires **no API key** and processes text embeddings offline to convert item names (like "Butter Chicken") into mathematical vectors. No specific command is needed to launch it; it boots automatically when you run the pipeline.

2. **Cloud API (Meal Coherence):**
   The project also utilizes Google's Gemini models natively to score cart "meal coherence". **You MUST provide a Gemini API Key to run this specific portion offline locally**.

**Configuring Gemini:**
1. Create a file named `.env` in the exact root of your repository (`FoodRecommenderSystem/.env`).
2. Add the following line:
   ```bash
   GEMINI_API_KEY="your-api-key-here"
   ```
*(Note: `.env` is inherently blocked by `.gitignore` so you will not accidentally upload your key to GitHub).*

---

## Single Command Execution (Recommended)

To run the entire simulated pipeline (Data Generation → Feature Engineering → Model Training → Meta-Learner → A/B Testing → Reporting), execute the orchestrator script:

```bash
python scripts/run_pipeline.py
```
> *Note: This will function entirely offline AFTER pinging Gemini for the initial text embeddings. Total execution time ranges from `15` to `35` minutes depending on your CPU density.*

---

## Step-by-Step Manual Breakdown

If you prefer to audit, debug, or execute steps surgically:

1. **Synthetic Data Generation (~35s)**
   ```bash
   python src/generate_synthetic_data.py
   ```
   Generates 7 discrete entity tables simulating an active food delivery platform.

2. **Compute LLM Vectors & Embeddings (~2m)**
   ```bash
   python src/llm_components.py
   ```
   Downloads `sentence-transformers` models locally and pings Gemini for text embeddings. You must run this before building the dataset.

3. **Flat Tensor Aggregation (~5-10m)**
   ```bash
   python src/build_training_dataset.py
   ```
   Melds interactions and cart sessions into a single flat `csao_training_data.csv`.

4. **Feature Engineering Pipeline (~10-15m)**
   ```bash
   python src/feature_pipeline.py
   ```
   Extracts 32 continuous and categorical temporal features.

5. **Base Ranker Training (~5m)**
   ```bash
   python src/train_base_models.py
   ```
   Trains LightGBM (via Optuna), Sequential GRU networks, and Item Collaborative Filtering arrays.

6. **Meta-Learner Fusion (~2m)**
   ```bash
   python src/train_meta_learner.py
   ```
   Fuses the outputs of the primary models using a Logistic Regressor.

7. **Validation & Evaluation (~1m)**
   ```bash
   python src/evaluate_model.py
   ```
   Calculates Precision/Recall, NDCG@K, evaluates business Lift impact, and commits charts to `reports/`.

---

##  Launching the Real-Time REST API

Initiate the async inference server locally:

```bash
uvicorn src.inference_service:app --host 0.0.0.0 --port 8000
```
- **API Swagger Sandbox**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Verification**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Launching the Web UI Dashboard

Testing raw JSON endpoints in a terminal isn't exciting. You can use the bespoke visual dashboard. **It does not require NodeJS or `npm run dev`**.

1. Locate the `ui/index.html` file in the project root.
2. Double-click the file to open it directly into any modern web browser (Chrome, Firefox, Safari) via `file:///...` protocol.
3. The vanilla HTML/JS application naturally polls your running `localhost:8000` uvicorn instance over standard HTTP requests. (Cors handles the connection).

**UI Features Accessible Locally**:
- Clicking the **Sun/Moon** icon toggles between Light and Dark mode rendering.
- Clicking the **Smartphone/Laptop** icon scales the application between a 480px mobile restricted view and a full-bleed horizontal desktop view natively.
- Placing an order will test the native **Track Order** real-time status UI.

---

## Hugging Face Spaces & Git LFS

This repository relies on **Git LFS (Large File Storage)** because the generated `csao_training_data.csv` frequently exceeds standard Git size limits (100MB+).

### Pushing changes to GitHub or HuggingFace:

1. **Ensure LFS is active:**
   ```bash
   git lfs install
   git lfs track "src/data/*.csv"
   git lfs track "data/*.csv"
   git add .gitattributes
   ```

2. **Hugging Face Secrets Config**
   If deploying to Hugging Face, remember to go to your Web UI, click `Settings` → `Variables and Secrets` and define `GEMINI_API_KEY`. Spaces cannot read your local `.env` file!

3. **Commit and deploy:**
   ```bash
   git add .
   git commit -m "Deploying latest application data"
   git push origin main
   ```

*(If you ever see a `pre-receive hook declined` error mentioning size, it means you missed tracking a `.csv` via `git lfs track`!)*

---

## Hard Reset & Cleanup Commands

If you need to instantly clear the database and refresh the state:

**Windows (PowerShell):**
```powershell
# Drops datasets and cached models completely
Remove-Item -Recurse -Force data\*, models\*, reports\*
```

**Linux / macOS / WSL:**
```bash
rm -rf data/* models/* reports/*
find . -type d -name "__pycache__" -exec rm -rf {} +
```

Then rebuild from scratch using `python scripts/run_pipeline.py`.

---

## Troubleshooting Glossary

| Symptom | Resolution / Fix |
|---|---|
| `ModuleNotFoundError: No module named 'dotenv'` | Virtual Environment is missing dependencies. Run `pip install -r requirements.txt`. |
| `GRU not loaded: No module named 'torch'` | System falls back gracefully to LightGBM. Can be ignored, or install PyTorch manually to resolve. |
| API returns `404` for UI carts | The selected dummy items don't natively exist on the specific restaurant's menu. Clear the cart in the UI. |
| UI Displays "Unknown" items | Stale dataset. Re-run `build_training_dataset.py`. |
| UI Displays "No Recommendations" | Ensure your terminal is running `uvicorn` and check Chrome/Edge DevTools (F12) for CORS or Network errors. |
