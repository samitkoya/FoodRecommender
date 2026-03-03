# ARCHITECTURE.md — CSAO Rail Recommendation System

## System Architecture Overview

The **Cart Super Add-On (CSAO)** Recommendation System is built on a highly optimized, two-stage inference architecture designed to return real-time recommendations in well under 200ms.

```text
┌───────────────┐     ┌───────────────────────┐     ┌──────────────────────────────────┐
│               │────>│                       │────>│                                  │
│   Web UI      │     │   FastAPI Server      │     │   Stacked Ensemble Ranker        │
│ (index.html)  │<────│   (< 200ms SLA)       │<────│                                  │
│               │     │                       │     │  Stage 1: Candidate Gen (~30ms)  │
└───────────────┘     └───────────┬───────────┘     │    ├── Co-occurrence Filter      │
                                  │                 │    ├── Meal Completion Rules     │
                      ┌───────────┴───────────┐     │    └── Popularity Fallback       │
                      │                       │     │                                  │
                      │   In-Memory Cache     │     │  Stage 2a: Base Rankers (||)     │
                      │  - User profiles      │     │    ├── LightGBM Ranker  (~8ms)   │
                      │  - Restaurant menus   │     │    ├── GRU Sequential   (~15ms)  │
                      │  - Co-occurrence      │     │    └── CF Scorer        (~5ms)   │
                      │  - Item embeddings    │     │         ↓ (parallel ≈ 15ms)      │
                      │  - Item popularity    │     │                                  │
                      │                       │     │  Stage 2b: Meta-Learner (~1ms)   │
                      └───────────────────────┘     │    LogisticRegression            │
                                                    │         ↓                        │
                                                    │  Diversity Filter + Re-rank      │
                                                    └──────────────────────────────────┘
```

## Data Lifecycle

The data pipeline runs through distinct phases before serving predictions online. Note that because of file size constraints, large assets leverage **Git LFS** and deploy to **Hugging Face Spaces**.

### Phase 1: Offline Data Engineering
- **Script**: `generate_synthetic_data.py`
  - Generates normalized tables (`users.csv`, `restaurants.csv`, `menu_items.csv`, `orders.csv`, `order_items.csv`, `cart_sessions.csv`, `csao_interactions.csv`).
- **Script**: `build_training_dataset.py`
  - Joins all tables into `csao_training_data.csv` (a single flat file containing all interactions and pre-computed stats).

### Phase 2: Offline Machine Learning
- **Script**: `feature_pipeline.py` builds the feature matrix.
- **Script**: `train_base_models.py` trains LightGBM, GRU (if enabled), and Collaborative Filtering matrix.
- **Script**: `train_meta_learner.py` trains the logistic regressor to dynamically weight the base rankers.
- **Script**: `evaluate_model.py` generates model reports and performance charts.

### Phase 3: Online Inference & Deployment
- Deployments to platforms like **Hugging Face Spaces** require tracking `data/` payloads via Git LFS.
- **Script**: `inference_service.py` runs the FastAPI web server.
- Serves traffic to the frontend Application via REST endpoints like `POST /v1/csao/recommend`.

##  Latency Budget Breakdown

### Fast Path (Early Exit)
Triggered when LightGBM returns highly confident scores. Serves ~40% of traffic.
| Component | Time Budget |
|---|---|
| Feature Retrieval | ~5ms |
| Candidate Generation | ~10ms |
| LightGBM Scoring | ~8ms |
| Filtering & Response | ~5ms |
| **Total Expected** | **~28ms** |

### Full Ensemble Path
Triggered when the system falls back to evaluating all models. Serves ~60% of traffic.
| Component | Time Budget |
|---|---|
| Feature Retrieval | ~5ms |
| Candidate Generation | ~10ms |
| Parallel Evaluation (LGB+GRU+CF) | ~15ms |
| Meta-Learner Scoring | ~1ms |
| Filtering & Response | ~5ms |
| **Total Expected** | **~36ms** |

## Critical Architectural Decisions

1. **Parallel Execution via Asyncio**: Wall-clock execution time during the ensemble phase equals the slowest model (usually GRU at ~15ms), rather than the sequential sum.
2. **Early Exits**: Checking LightGBM confidence thresholds stops unnecessary downstream model evaluation, saving heavy computational resources.
3. **Temporal Data Splitting**: Validation and Testing splits strictly respect chronological order, simulating real-world production constraints and preventing data leakage.
4. **Offline LLMs, Online ML**: Tasks like Semantic Embeddings and Gemini Meal Coherence are batched and executed strictly offline, eliminating API jitter from the real-time serving path.
5. **Git LFS over Cloud Buckets**: To keep the project monolithically portable across generic hosting platforms like Hugging Face without requiring complex IAM authentication to AWS S3/GCS.
