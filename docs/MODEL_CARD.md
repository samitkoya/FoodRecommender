# MODEL_CARD.md — CSAO Recommendation System

## Model Overview
- **Name**: CSAO Stacked Ensemble (`v2.1`)
- **Architecture Type**: Two-stage recommendation pipeline (candidate generation followed by a stacked meta-ensemble)
- **Primary Task**: Predict which restaurant add-on items a user will accept based on their current cart state, profile, and historical data.
- **Ensemble Composition**: Base rankers comprising LightGBM, GRU (Sequential), and Item-Item CF, weighted dynamically by a Logistic Regression meta-learner.
- **Deployment**: Configured for edge deployment or cloud inference (e.g. Hugging Face Spaces).

## Base Rankers Detailed

### 1. LightGBM LambdaRank
- **Objective Function**: `lambdarank` (optimized natively for NDCG).
- **Inputs**: 32 engineered tabular features covering user state, cart complexity, candidate item properties, meal completion logic, and co-occurrence statistics.
- **Hyperparameter Optimization**: Bayesian optimization via Optuna.
- **Characteristics**: Fast inference speeds (~8ms). Exceptionally strong at navigating complex feature interactions.

### 2. GRU Sequential Ranker
- **Architecture Flow**: Embedding Space(64) → GRU(128) → Linear Layer(64) → Sigmoid Output.
- **Inputs**: Ordered sequence of items added historically to the cart array.
- **Characteristics**: Highly effective at capturing session dynamics and sequential trends. Degrades significantly during cold-start scenarios or sparse user history.

### 3. Collaborative Filtering (CF) Scorer
- **Methodology**: In-memory co-occurrence matrix derived from order history cross-sections.
- **Characteristics**: Extremely high fidelity for globally popular combinations (e.g., Coke and Pizza). Weak for newly cataloged items.

### 4. Meta-Learner Layer
- **Methodology**: Logistic Regression (`C=1.0`).
- **Inputs**: Raw scores from base rankers (`score_lgb`, `score_gru`, `score_cf`), combined with contextual heuristics (`user_segment`, `meal_time`, `is_cold_start`, `cart_size`, `completeness`).
- **Training Guardrail**: Trained *exclusively* on the held-out validation set to prevent ensemble overfitting.

## Training Dataset Characteristics
- **Data Source**: Synthetically generated dataset mirroring typical food delivery platforms.
- **Dataset Scale**: 10,000 users, 500 restaurants, ~12,000 items, and 100,000 shopping sessions.
- **Data Partitions**: Strict Temporal Splits (60% Train / 20% Validation / 20% Test).
- **Target Variable**: `was_accepted` (Binary classification representing whether a user added the recommended item to their cart).
- **Label Imbalance**: Positive hit rate varies between ~15-25%.
- **Storage Strategy**: Core datasets are versioned using **Git Large File Storage (LFS)**.

## Performance Evaluation
- **Primary Metrics**: AUC-ROC, NDCG@K (K=3,5,10), Precision@K, Recall@K
- **Segmented Analysis**: Models are independently evaluated across `user_segment`, `meal_time_slot`, `cart_size`, and strict `cold_start` domains to ensure equitable coverage.

## Known Limitations and Tradeoffs
1. **Synthetic Generation Artifacts**: The model is trained on synthetic properties. Distributions will naturally drift when subjected to live, organic production data.
2. **Cold-Start Bias**: Brand new users implicitly default to popularity-based fallback models, lowering personalized relevance initially.
3. **Diversity Guardrails**: The diversity filter operates on top-level categorization (Main/Side/Beverage) rather than semantic intent, which may still result in flavor clashes.
4. **Static Co-occurrence**: The CF matrix is static between offline batch runs (e.g., weekly), meaning abrupt viral trends will not be captured instantaneously.

## Best Practices and Intended Use
- **Use Case**: Embedded directly into the checkout or "cart builder" flow of high-volume delivery applications.
- **Latency SLA**: Real-time evaluation meant to clear entirely under 200ms end-to-end.
- **Ethical Considerations**: Explicit diversity penalties prevent the system from exclusively surfacing high-margin items over relevant customer choices. User tracking focuses on contextual behaviors rather than PII.
