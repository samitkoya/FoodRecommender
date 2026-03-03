---
title: CSAO Food Recommender
emoji: 🍽️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# CSAO (Cart Super Add-On) Recommendation System

> **An intelligent, real-time recommendation engine that suggests the perfect complementary food items as customers build their delivery carts.**

[![Live Demo on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/samitkoya/FoodRecommender)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-ff69b4)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c)

When a customer adds **Chicken Biryani** to their cart, the system dynamically suggests **Raita** → then **Gulab Jamun** → then **Cold Drink**. As the cart state changes, the recommendations adapt instantly.

🔗 **Live Deployment - https://huggingface.co/spaces/samitkoya/FoodRecommender**

---

## What Is This Project?

Imagine ordering on **Zomato**. After you add **Chicken Biryani** to your cart, a small tray appears stating: *"Hey, want some Raita or a Thums Up with that?"* — that cross-sell mechanism is called the **CSAO Rail** (**C**art **S**uper **A**dd-**O**n).

This repository builds the complete machine learning **brain** powering that rail. It achieves three core priorities:
1. **Candidate Search**: What items in this restaurant *could* pair with the cart?
2. **Ranking**: Which of those candidate items is *this specific user* most likely to actually want?
3. **Delivery**: Serve the best options sequentially, under an extreme **200 millisecond** Service Level Agreement (SLA).

---

## How It Works (The 30-Second Summary)

```text
[Customer adds "Biryani" to cart]
             ↓
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Candidate Generation (~30ms)                   │
│   "What items MIGHT go with Biryani?"                   │
│   → Filters by co-occurrence, meal rules, popularity    │
├─────────────────────────────────────────────────────────┤
│ Stage 2: Parallel Ranking (~15ms)                       │  
│   "Which items will THIS specific user want?"           │
│   → 3 ML models race in PARALLEL:                       │
│       1. LightGBM (Tabular Features)       → Score: 0.72│
│       2. GRU Network (Cart Sequence)       → Score: 0.68│
│       3. Collaborative Filtering (History) → Score: 0.61│
│                                                         │
│   → Meta-Learner Layer dynamically synthesizes scores   │
├─────────────────────────────────────────────────────────┤
│ Stage 3: Diversification (~2ms)                         │
│   → Ensures categorical balance (e.g., max 3 beverages) │
└─────────────────────────────────────────────────────────┘
             ↓
[API returns: Raita, Salan, Coke, Gulab Jamun...]
Total latency: ~35-45ms (well under 200ms strict SLA)
```

---

## System Architecture

```text
Web UI (HTML/JS)  ──→  FastAPI Server (< 200ms SLA)  ──→  Ensemble Engine
                                    │
                            In-Memory Cache
                     (users, menus, embeddings)
                                    │
              ┌─────────────────────┼─────────────────────┐
              ↓                     ↓                     ↓
     Candidate Generation    Base Rankers (parallel)   Meta-Learner
      - Co-occurrence        - LightGBM   (~8ms)      - LogisticRegression
      - Meal Completion      - GRU        (~15ms)       combines scores
      - Popularity           - CF Scorer  (~5ms)           ↓
                                                    Diversity Filter
                                                     + Re-ranking
```

---

## ML Model Details

The system employs a **stacked meta-ensemble** (`v2.1`) consisting of three specialized base rankers fused by a meta-learner:

| Base Ranker | Architecture | Strength | Latency |
|---|---|---|---|
| **LightGBM** | LambdaRank (Optuna-tuned) | Tabular feature interactions (32 engineered features) | ~8ms |
| **GRU Sequential** | Embedding(64) → GRU(128) → Linear(64) → Sigmoid | Chronological cart sequence dynamics | ~15ms |
| **Collaborative Filtering** | In-memory co-occurrence matrix | Globally popular item combinations | ~5ms |
| **Meta-Learner** | Logistic Regression (C=1.0) | Dynamically weights base ranker outputs per context | ~1ms |

**Early Exit Circuit Breaker**: When LightGBM returns confidence > 85%, the server immediately skips the slower neural networks and returns the result directly (~40% of traffic).

### Training Data
- **Source**: Synthetically generated dataset mirroring food delivery platforms.
- **Scale**: 10,000 users, 500 restaurants, ~12,000 items, and 100,000 shopping sessions.
- **Splits**: Strict Temporal Partitioning (60% Train / 20% Validation / 20% Test) preventing data leakage.
- **Target**: `was_accepted` — binary classification of whether a user added the recommended item.

### Key Metrics (Simulated Synthetic Data)

| Model Pathway | AUC-ROC | NDCG@5 | NDCG@10 | Time Cost |
|---|---|---|---|---|
| Random Baseline | ~0.50 | ~0.08 | ~0.10 | ~2ms |
| Popularity Baseline | ~0.55 | ~0.12 | ~0.14 | ~5ms |
| Collaborative Filtering| ~0.63 | ~0.17 | ~0.19 | ~5ms |
| GRU Sequential | ~0.63 | ~0.18 | ~0.20 | ~15ms |
| **LightGBM (Solo)** | **~0.71** | **~0.23** | **~0.25** | **~8ms** |
| **Full Stacked Ensemble**| **~0.75** | **~0.27** | **~0.30** | **~35ms** |

> *Metrics derived from synthetic order datasets. Production deployments trained on genuine proprietary data will yield substantially different results.*

### Latency Budget

| Path | Scenario | Total |
|---|---|---|
| **Fast Path (Early Exit)** | LightGBM highly confident (~40% of traffic) | **~28ms** |
| **Full Ensemble** | All models evaluated in parallel (~60% of traffic) | **~36ms** |

---

## Project Structure

```text
FoodRecommenderSystem/
│
├── ui/                                ← Vanilla Web UI testing dashboard (HTML/JS)
├── requirements.txt                   ← Python dependencies (FastAPI, Torch, LGBM, etc.)
├── Dockerfile                         ← HuggingFace Spaces Docker deployment config
├── README.md                          ← You are here!
│
├── src/                               ← Core Python modules
│   ├── generate_synthetic_data.py     ← Step 1: Synthesizes users, restaurants, orders
│   ├── build_training_dataset.py      ← Step 2: Joins into flat CSV
│   ├── feature_pipeline.py            ← Step 3: Engineers 32 temporal features
│   ├── train_base_models.py           ← Step 4: Trains LGBM + GRU + CF
│   ├── train_meta_learner.py          ← Step 5: Trains ensemble logistical combiner
│   ├── evaluate_model.py              ← Step 6: Full validation + metric charts
│   ├── cold_start_pipeline.py         ← Step 7: Zero-history user fallback handling
│   ├── ab_test_analysis.py            ← Step 8: Statistical modeling framework
│   ├── llm_components.py             ← Step 9: LLM text embeddings computation
│   ├── ensemble_inference.py          ← Serving: Core parallel ensemble math
│   └── inference_service.py           ← Serving: FastAPI real-time REST server
│
├── config/
│   └── settings.py                    ← Global configs, ML hyperparams, constants
│
├── scripts/
│   ├── run_pipeline.py                ← Single-click script running Steps 1-8
│   └── test_api.py                    ← API validation tests
│
├── docs/                              ← In-depth Documentation
│   ├── EXPLAINED.md                   ← Beginner-friendly architecture walkthrough
│   ├── HOW_TO_RUN.md                  ← Step-by-step execution & deployment guide
│   ├── ARCHITECTURE.md                ← System design diagrams + latency budget
│   └── MODEL_CARD.md                  ← ML details, metrics, and limitations
│
├── data/                              ← Generated datasets (uses Git LFS for large CSVs)
├── models/                            ← Serialized artifacts (.pkl, .pt)
└── reports/                           ← Saved performance plots and test logs
```

---

## Quick Start (Local & Offline Execution)

### Prerequisites
- **Python 3.10+**
- `~1GB` free disk space for data generation
- **Git LFS** installed (for handling large datasets)
- **PyTorch** (optional) — the GRU component gracefully bypasses if unavailable

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure LLM API Keys

The project uses two LLM components during offline dataset creation:

| Component | Type | API Key Required? |
|---|---|---|
| **Semantic Embeddings** (`sentence-transformers`) | Local LLM | ❌ No — runs entirely on CPU |
| **Meal Coherence** (Google Gemini) | Cloud API | ✅ Yes — requires `GEMINI_API_KEY` |

Create a `.env` file in the project root:
```bash
GEMINI_API_KEY="your-api-key-here"
```
> *`.env` is blocked by `.gitignore` — your key will never be committed to source control.*

### 3. Compute LLM Vectors & Embeddings
```bash
python src/llm_components.py
```
> *Downloads open-source embedding models locally*

### 4. Run the Full ML Pipeline
```bash
python scripts/run_pipeline.py
```
This single command executes: Data Generation → Feature Engineering → Model Training → Meta-Learner → A/B Testing → Reporting.
> *Total execution: ~15-35 minutes depending on CPU.*

### 5. Start the API Server
```bash
uvicorn src.inference_service:app --host 0.0.0.0 --port 8000
```
- **Swagger Sandbox**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Launching the Web UI Dashboard

The interactive dashboard requires **zero build tools** — no Node.js, no `npm`.

1. Open `ui/index.html` directly in any modern browser (Chrome, Firefox, Edge).
2. It automatically communicates with your running `localhost:8000` API via REST.
3. **Built-in UI Toggles**:
   - **Dark/Light Mode**: Click the Sun/Moon icon.
   - **Mobile/Desktop View**: Click the Smartphone/Laptop icon to scale between 480px mobile and full-bleed desktop layouts.
   - **Order Tracking**: Place an order to test the real-time status tracking UI.

---

## API Usage

### `POST /v1/csao/recommend`

**Request:**
```json
{
  "user_id": "u_100",
  "restaurant_id": "r_3",
  "cart_items": ["i_50"],
  "n_recommendations": 10
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "item_id": "i_55", 
      "item_name": "Restaurant 3 Raita", 
      "category": "Side",
      "price": 70.0, 
      "score": 0.87, 
      "rank": 1, 
      "reason": "side", 
      "is_veg": true
    },
    {
      "item_id": "i_60", 
      "item_name": "Restaurant 3 Coke", 
      "category": "Beverage",
      "price": 40.0, 
      "score": 0.81, 
      "rank": 2, 
      "reason": "beverage", 
      "is_veg": true
    }
  ],
  "latency_ms": 28.5,
  "is_cold_start": false,
  "ensemble_path": "early_exit_lgb"
}
```

---

## The Pipeline: Step-by-Step Breakdown

<details>
<summary><strong>Click to expand the full manual pipeline</strong></summary>

If you prefer to audit, debug, or execute steps individually:

| Step | Script | Description | Runtime |
|---|---|---|---|
| 1 | `python src/generate_synthetic_data.py` | Generates 7 entity tables (users, restaurants, orders, etc.) | ~35s |
| 2 | `python src/llm_components.py` | Downloads local sentence-transformers + Gemini embeddings | ~2m |
| 3 | `python src/build_training_dataset.py` | Aggregates all tables into flat `csao_training_data.csv` | ~5-10m |
| 4 | `python src/feature_pipeline.py` | Engineers 32 continuous/categorical temporal features | ~10-15m |
| 5 | `python src/train_base_models.py` | Trains LightGBM (Optuna), GRU, and CF models | ~5m |
| 6 | `python src/train_meta_learner.py` | Trains LogisticRegression meta-learner on validation set | ~2m |
| 7 | `python src/evaluate_model.py` | Calculates metrics and generates charts to `reports/` | ~1m |

</details>

---

## Known Limitations

- **Synthetic Training Data**: Model behavior will naturally drift when subjected to live organic production data.
- **Cold-Start Bias**: New users default to popularity-based fallback, lowering initial personalization.
- **Static Co-occurrence Matrix**: The CF matrix is refreshed only during batch runs, meaning viral trends won't be captured instantly.
- **Category-Level Diversity**: The diversity filter operates on top-level categories rather than semantic intent, which may still result in flavor clashes.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Primary Base Ranker** | LightGBM (`lambdarank`) | Tabular data features (fastest, most accurate) |
| **Sequential Base Ranker** | PyTorch (GRU) | Learns from chronological cart addition sequence |
| **Collaborative Filtering** | Scikit-learn | In-memory item-item co-occurrence scoring |
| **Meta-Learner** | Logistic Regression | Dynamically fuses base ranker outputs |
| **API Server** | FastAPI & Uvicorn | Ultra-fast async real-time routing |
| **Hyperparameter Tuning** | Optuna | Bayesian hyperparameter optimization |
| **LLM Embeddings** | Sentence-transformers | Item text embeddings for cold-start logic |
| **Meal Coherence** | Google Gemini API | Offline meal pairing scoring |
| **Data & Stats** | Pandas, NumPy, Scipy | Core math, evaluation, and pipeline assembly |
| **Visualization** | Matplotlib, Seaborn | Performance charts and evaluation plots |
| **Web Frontend** | Vanilla HTML/JS | Dependency-free interactive UI dashboard |

---

## Hugging Face Spaces Deployment

🔗 **Live Demo**: [https://huggingface.co/spaces/samitkoya/FoodRecommender](https://huggingface.co/spaces/samitkoya/FoodRecommender)

This repository is fully configured for deployment on **Hugging Face Spaces** using a `Dockerfile` SDK.

---

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `ModuleNotFoundError: No module named 'dotenv'` | Run `pip install -r requirements.txt` to ensure all dependencies are installed. |
| `GRU not loaded: No module named 'torch'` | System falls back gracefully to LightGBM. Install PyTorch manually to resolve, or ignore. |
| API returns `404` for UI carts | Selected dummy items don't exist on the restaurant's menu. Clear the cart in the UI. |
| UI displays "Unknown" items | Stale dataset. Re-run `python src/build_training_dataset.py`. |
| UI displays "No Recommendations" | Ensure `uvicorn` is running. Check DevTools (F12) for CORS or network errors. |

---

## Documentation

For deeper technical exploration, refer to the docs:

| Document | Description |
|---|---|
| [EXPLAINED.md](docs/EXPLAINED.md) | Beginner-friendly walkthrough of the entire project in plain English |
| [HOW_TO_RUN.md](docs/HOW_TO_RUN.md) | Comprehensive step-by-step execution and deployment guide |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design diagrams and latency budget breakdowns |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | ML model specifications, metrics, and known limitations |
