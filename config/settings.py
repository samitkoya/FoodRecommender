"""
Centralized configuration for the CSAO Recommendation System.
All constants, paths, encoding maps, hyperparameters, and model configs.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# ── Scale ──
N_USERS = 10_000
N_RESTAURANTS = 500
N_ITEMS_APPROX = 5_000
N_ORDERS_APPROX = 500_000

# ── Geography ──
CITIES = ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Pune"]

# ── Segments ──
USER_SEGMENTS = ["budget", "regular", "premium"]
PRICE_RANGES = ["budget", "mid", "premium"]
CATEGORIES = ["Main", "Side", "Beverage", "Dessert", "Starter"]
MEAL_TIMES = ["breakfast", "lunch", "evening_snack", "dinner", "late_night"]

# ── Encoding Maps ──
SEG_MAP = {"budget": 0, "regular": 1, "premium": 2}
PRICE_MAP = {"budget": 0, "mid": 1, "premium": 2}
CAT_MAP = {"Main": 0, "Side": 1, "Beverage": 2, "Dessert": 3, "Starter": 4}
MEAL_MAP = {"breakfast": 0, "lunch": 1, "evening_snack": 2, "dinner": 3, "late_night": 4}

# ── Feature Columns (used by all models and inference) ──
FEATURE_COLS = [
    # User features
    "user_segment_enc", "user_order_frequency", "user_avg_order_value",
    "user_addon_acceptance_rate", "user_price_sensitivity", "days_since_last_order",
    "is_cold_start_user",
    # Cart context
    "cart_total_value", "cart_item_count", "cart_avg_item_price",
    "cart_is_single_item", "candidate_price_pct_of_cart",
    # Meal completion
    "meal_has_main", "meal_has_side", "meal_has_beverage", "meal_has_dessert",
    "meal_completeness_score", "candidate_fills_gap", "cuisine_coherence_score",
    # Candidate item
    "item_category_enc", "item_price", "item_is_veg",
    "item_popularity_rank", "item_avg_rating", "item_attachment_rate",
    "co_occurrence_score",
    # Context
    "hour_of_day", "day_of_week", "meal_time_enc", "is_weekend",
    # Cold start
    "is_cold_start_item",
    # Sequential
    "sequential_transition_score",
]

# ── LightGBM Defaults ──
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [3, 5, 10],
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}

NUM_BOOST_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 50
N_OPTUNA_TRIALS = 20

# ── GRU Model ──
GRU_CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 128,
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 15,
    "dropout": 0.2,
}

# ── Ensemble ──
EARLY_EXIT_THRESHOLD = 0.85

CIRCUIT_BREAKER_CONFIG = {
    "lgb": {"max_latency_ms": 15, "fallback": "popularity"},
    "gru": {"max_latency_ms": 25, "fallback": "lgb_only"},
    "cf":  {"max_latency_ms": 10, "fallback": "lgb_only"},
}

# ── Cold Start Thresholds ──
COLD_START_USER_ORDER_THRESHOLD = 3
COLD_START_ITEM_INTERACTION_THRESHOLD = 50

# ── API ──
API_HOST = "0.0.0.0"
API_PORT = 8000
DEFAULT_TOP_N = 10
