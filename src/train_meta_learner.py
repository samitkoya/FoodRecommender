"""
train_meta_learner.py — Train Stacking Meta-Learner for CSAO Ensemble

Generates base model predictions on the HELD-OUT VALIDATION set,
then trains a LogisticRegression meta-learner that learns
context-dependent model weights (e.g., trust GRU more for 3+ item carts).

Run: python train_meta_learner.py
"""

import os, pickle, json, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLS = [
    "user_segment_enc", "user_order_frequency", "user_avg_order_value",
    "user_addon_acceptance_rate", "user_price_sensitivity", "days_since_last_order",
    "is_cold_start_user",
    "cart_total_value", "cart_item_count", "cart_avg_item_price",
    "cart_is_single_item", "candidate_price_pct_of_cart",
    "meal_has_main", "meal_has_side", "meal_has_beverage", "meal_has_dessert",
    "meal_completeness_score", "candidate_fills_gap", "cuisine_coherence_score",
    "item_category_enc", "item_price", "item_is_veg",
    "item_popularity_rank", "item_avg_rating", "item_attachment_rate",
    "co_occurrence_score",
    "hour_of_day", "day_of_week", "meal_time_enc", "is_weekend",
    "is_cold_start_item",
    "sequential_transition_score",
]


def get_lgb_scores(val_df, feature_cols):
    """Get LightGBM predictions on validation set."""
    print("  Getting LightGBM scores...")
    with open(os.path.join(MODEL_DIR, "lgb_model.pkl"), "rb") as f:
        model = pickle.load(f)
    X = val_df[feature_cols].fillna(0)
    return model.predict(X)


def get_gru_scores(val_df):
    """Get GRU predictions on validation set."""
    print("  Getting GRU scores...")
    try:
        import torch
        import torch.nn as nn

        with open(os.path.join(MODEL_DIR, "gru_item2idx.pkl"), "rb") as f:
            item2idx = pickle.load(f)

        vocab_size = len(item2idx) + 1

        class GRUCartEncoder(nn.Module):
            def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, dropout=0.2):
                super().__init__()
                self.item_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
                self.scorer = nn.Sequential(
                    nn.Linear(hidden_dim + embed_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )

            def forward(self, cart_seq, candidate_id):
                cart_emb = self.item_embedding(cart_seq)
                _, cart_state = self.gru(cart_emb)
                cart_state = cart_state.squeeze(0)
                cand_emb = self.item_embedding(candidate_id)
                combined = torch.cat([cart_state, cand_emb], dim=-1)
                return torch.sigmoid(self.scorer(combined)).squeeze(-1)

        device = torch.device("cpu")
        model = GRUCartEncoder(vocab_size).to(device)

        gru_path = os.path.join(MODEL_DIR, "gru_model.pt")
        if os.path.exists(gru_path):
            model.load_state_dict(torch.load(gru_path, map_location=device, weights_only=True))
        else:
            print("    GRU model not found, using random scores.")
            return np.random.random(len(val_df))

        model.eval()
        max_seq = 10
        scores = []

        for _, row in val_df.iterrows():
            try:
                cart = json.loads(row["cart_state_at_recommendation"]) if pd.notna(row["cart_state_at_recommendation"]) else []
            except:
                cart = []

            cart_idx = [item2idx.get(str(c), 0) for c in cart]
            if len(cart_idx) == 0:
                cart_idx = [0]
            cart_idx = cart_idx[-max_seq:]
            while len(cart_idx) < max_seq:
                cart_idx.insert(0, 0)

            cand_idx = item2idx.get(str(row["recommended_item_id"]), 0)

            with torch.no_grad():
                cart_t = torch.tensor([cart_idx], dtype=torch.long)
                cand_t = torch.tensor([cand_idx], dtype=torch.long)
                pred = model(cart_t, cand_t).item()
            scores.append(pred)

        return np.array(scores)

    except Exception as e:
        print(f"    GRU scoring failed: {e}")
        return np.random.random(len(val_df)) * 0.5


def get_cf_scores(val_df):
    """Get CF scores on validation set."""
    print("  Getting CF scores...")
    with open(os.path.join(MODEL_DIR, "co_occurrence_matrix.pkl"), "rb") as f:
        co_matrix = pickle.load(f)

    scores = []
    for _, row in val_df.iterrows():
        try:
            cart = json.loads(row["cart_state_at_recommendation"]) if pd.notna(row["cart_state_at_recommendation"]) else []
        except:
            cart = []

        candidate = row["recommended_item_id"]
        if not cart:
            scores.append(0.0)
        else:
            s = [co_matrix.get((c, candidate), 0) for c in cart]
            scores.append(float(np.mean(s)) if s else 0.0)

    return np.array(scores)


def main():
    print("=" * 55)
    print("  CSAO META-LEARNER TRAINING")
    print("=" * 55)

    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_features.csv"))
    feature_cols = [c for c in FEATURE_COLS if c in val_df.columns]

    print(f"  Validation set: {len(val_df)} rows")

    # Get base model scores
    score_lgb = get_lgb_scores(val_df, feature_cols)
    score_gru = get_gru_scores(val_df)
    score_cf = get_cf_scores(val_df)

    # Build meta-features
    print("  Building meta-features...")
    meta_features = pd.DataFrame({
        "score_lgb": score_lgb,
        "score_gru": score_gru[:len(val_df)],
        "score_cf": score_cf,
        "user_segment_enc": val_df["user_segment_enc"].fillna(1).values,
        "meal_time_enc": val_df["meal_time_enc"].fillna(3).values,
        "is_cold_start_user": val_df["is_cold_start_user"].fillna(0).values,
        "cart_item_count": val_df["cart_item_count"].fillna(1).values,
        "meal_completeness_score": val_df["meal_completeness_score"].fillna(0).values,
    })

    y_val = val_df["was_accepted"].values

    # Train meta-learner
    print("  Training LogisticRegression meta-learner...")
    meta_learner = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
    meta_learner.fit(meta_features.fillna(0), y_val)

    # Evaluate
    meta_preds = meta_learner.predict_proba(meta_features.fillna(0))[:, 1]
    try:
        auc = roc_auc_score(y_val, meta_preds)
    except:
        auc = 0.5

    print(f"\n  Meta-learner AUC on validation: {auc:.4f}")
    print(f"  Coefficients:")
    for name, coef in zip(meta_features.columns, meta_learner.coef_[0]):
        print(f"    {name:30s}: {coef:+.4f}")
    print(f"  Intercept: {meta_learner.intercept_[0]:+.4f}")

    # Compare: meta-learner vs individual models
    print("\n  --- Comparison on Validation Set ---")
    for name, scores in [("LightGBM", score_lgb), ("GRU", score_gru[:len(val_df)]),
                         ("CF", score_cf), ("Meta-Learner", meta_preds)]:
        try:
            a = roc_auc_score(y_val, scores)
        except:
            a = 0.5
        print(f"    {name:20s} AUC={a:.4f}")

    # Save
    with open(os.path.join(MODEL_DIR, "meta_learner.pkl"), "wb") as f:
        pickle.dump(meta_learner, f)

    print(f"\n  Meta-learner saved to {MODEL_DIR}/meta_learner.pkl")
    print("  Training complete!")


if __name__ == "__main__":
    main()
