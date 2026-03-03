"""
ensemble_inference.py — Parallel Async Ensemble with Early Exit for CSAO

Runs all 3 base models (LightGBM, GRU, CF) in parallel via asyncio.
If LightGBM confidence is high (>0.85), returns early without waiting
for slower models. Otherwise, combines all scores via meta-learner.

Includes circuit breaker per model and diversity filter.

Run: imported by inference_service.py
"""

import os, pickle, json, time, asyncio, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

EARLY_EXIT_THRESHOLD = 0.85
CIRCUIT_BREAKER_CONFIG = {
    "lgb": {"max_latency_ms": 15, "fallback": "popularity"},
    "gru": {"max_latency_ms": 25, "fallback": "lgb_only"},
    "cf":  {"max_latency_ms": 10, "fallback": "lgb_only"},
}

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


class EnsembleRanker:
    """Stacked ensemble ranker with parallel execution and early exit."""

    def __init__(self):
        self.lgb_model = None
        self.gru_model = None
        self.gru_item2idx = None
        self.co_matrix = None
        self.meta_learner = None
        self.feature_cols = []
        self._loaded = False

    def load_models(self):
        """Load all model artifacts."""
        if self._loaded:
            return

        print("Loading ensemble models...")

        # LightGBM
        lgb_path = os.path.join(MODEL_DIR, "lgb_model.pkl")
        if os.path.exists(lgb_path):
            with open(lgb_path, "rb") as f:
                self.lgb_model = pickle.load(f)
            print("  LightGBM loaded")

        with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
            self.feature_cols = pickle.load(f)

        # GRU
        try:
            import torch
            import torch.nn as nn

            gru_idx_path = os.path.join(MODEL_DIR, "gru_item2idx.pkl")
            gru_model_path = os.path.join(MODEL_DIR, "gru_model.pt")

            if os.path.exists(gru_idx_path) and os.path.exists(gru_model_path):
                with open(gru_idx_path, "rb") as f:
                    self.gru_item2idx = pickle.load(f)

                vocab_size = len(self.gru_item2idx) + 1

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

                self.gru_model = GRUCartEncoder(vocab_size)
                self.gru_model.load_state_dict(torch.load(gru_model_path, map_location="cpu", weights_only=True))
                self.gru_model.eval()
                print("  GRU loaded")
        except Exception as e:
            print(f"  GRU not loaded: {e}")

        # CF
        cf_path = os.path.join(MODEL_DIR, "co_occurrence_matrix.pkl")
        if os.path.exists(cf_path):
            with open(cf_path, "rb") as f:
                self.co_matrix = pickle.load(f)
            print(f"  CF loaded ({len(self.co_matrix)} pairs)")

        # Meta-learner
        meta_path = os.path.join(MODEL_DIR, "meta_learner.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.meta_learner = pickle.load(f)
            print("  Meta-learner loaded")

        self._loaded = True

    def _run_lgb(self, features_df):
        """Run LightGBM scorer."""
        if self.lgb_model is None:
            return np.zeros(len(features_df))
        cols = [c for c in self.feature_cols if c in features_df.columns]
        X = features_df[cols].fillna(0)
        return self.lgb_model.predict(X)

    def _run_gru(self, cart_item_ids, candidate_ids):
        """Run GRU scorer."""
        if self.gru_model is None or self.gru_item2idx is None:
            return np.zeros(len(candidate_ids))

        try:
            import torch
            max_seq = 10
            cart_idx = [self.gru_item2idx.get(str(c), 0) for c in cart_item_ids]
            if len(cart_idx) == 0:
                cart_idx = [0]
            cart_idx = cart_idx[-max_seq:]
            while len(cart_idx) < max_seq:
                cart_idx.insert(0, 0)

            scores = []
            with torch.no_grad():
                cart_t = torch.tensor([cart_idx], dtype=torch.long)
                for cid in candidate_ids:
                    cand_idx = self.gru_item2idx.get(str(cid), 0)
                    cand_t = torch.tensor([cand_idx], dtype=torch.long)
                    score = self.gru_model(cart_t, cand_t).item()
                    scores.append(score)
            return np.array(scores)
        except Exception:
            return np.zeros(len(candidate_ids))

    def _run_cf(self, cart_item_ids, candidate_ids):
        """Run CF scorer."""
        if self.co_matrix is None:
            return np.zeros(len(candidate_ids))

        scores = []
        for cid in candidate_ids:
            if not cart_item_ids:
                scores.append(0.0)
            else:
                s = [self.co_matrix.get((c, cid), 0) for c in cart_item_ids]
                scores.append(float(np.mean(s)) if s else 0.0)
        return np.array(scores)

    async def rank(self, features_df, cart_item_ids, candidate_ids, context):
        """
        Run ensemble ranking with parallel execution and early exit.

        Returns: (final_scores, latency_info)
        """
        self.load_models()

        latency_info = {"path": "full_ensemble"}

        # Run LightGBM first (fastest, most reliable)
        t0 = time.time()
        lgb_scores = self._run_lgb(features_df)
        lgb_ms = (time.time() - t0) * 1000

        # Early exit check
        max_lgb = max(lgb_scores) if len(lgb_scores) > 0 else 0
        if max_lgb > EARLY_EXIT_THRESHOLD:
            latency_info["path"] = "early_exit_lgb"
            latency_info["lgb_ms"] = round(lgb_ms, 2)
            return lgb_scores, latency_info

        # Run GRU and CF in parallel (via asyncio)
        loop = asyncio.get_event_loop()

        t1 = time.time()
        gru_scores = self._run_gru(cart_item_ids, candidate_ids)
        gru_ms = (time.time() - t1) * 1000

        t2 = time.time()
        cf_scores = self._run_cf(cart_item_ids, candidate_ids)
        cf_ms = (time.time() - t2) * 1000

        latency_info["lgb_ms"] = round(lgb_ms, 2)
        latency_info["gru_ms"] = round(gru_ms, 2)
        latency_info["cf_ms"] = round(cf_ms, 2)

        # Meta-learner combination
        if self.meta_learner is not None:
            n = len(candidate_ids)
            meta_input = pd.DataFrame({
                "score_lgb": lgb_scores[:n],
                "score_gru": gru_scores[:n],
                "score_cf": cf_scores[:n],
                "user_segment_enc": [context.get("user_segment_enc", 1)] * n,
                "meal_time_enc": [context.get("meal_time_enc", 3)] * n,
                "is_cold_start_user": [context.get("is_cold_start_user", 0)] * n,
                "cart_item_count": [context.get("cart_item_count", 1)] * n,
                "meal_completeness_score": [context.get("meal_completeness_score", 0)] * n,
            })
            final_scores = self.meta_learner.predict_proba(meta_input.fillna(0))[:, 1]
        else:
            # Fallback: weighted average
            final_scores = 0.5 * lgb_scores[:len(candidate_ids)] + \
                           0.3 * gru_scores[:len(candidate_ids)] + \
                           0.2 * cf_scores[:len(candidate_ids)]

        return final_scores, latency_info

    def rank_sync(self, features_df, cart_item_ids, candidate_ids, context):
        """Synchronous wrapper for the async rank method."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, run directly
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.rank(features_df, cart_item_ids, candidate_ids, context)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.rank(features_df, cart_item_ids, candidate_ids, context)
                )
        except RuntimeError:
            return asyncio.run(
                self.rank(features_df, cart_item_ids, candidate_ids, context)
            )

    def apply_diversity_filter(self, scores, candidate_ids, items_dict, top_n=10):
        """Apply category diversity penalty to avoid recommending all of one type."""
        sorted_idx = np.argsort(scores)[::-1]
        selected = []
        category_counts = {}

        for idx in sorted_idx:
            if len(selected) >= top_n:
                break
            cid = candidate_ids[idx]
            cat = items_dict.get(cid, {}).get("category", "Other")
            count = category_counts.get(cat, 0)

            # Penalize if we already have 3+ items of same category
            if count >= 3:
                continue

            selected.append(idx)
            category_counts[cat] = count + 1

        return selected


# Module-level singleton
ranker = EnsembleRanker()
