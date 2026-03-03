"""
train_base_models.py — Train 3 Independent Base Models for CSAO Stacked Ensemble

  1. LightGBM LambdaRank — handles tabular features + interactions
  2. GRU Sequential Model (PyTorch) — captures cart item addition order
  3. Item-Item CF Scorer — collaborative signal from co-occurrence

Each model is trained independently and evaluated with NDCG@K, Precision@K, Recall@K.
Saves: lgb_model.pkl, gru_model.pt, co_occurrence_matrix.pkl

Run: python train_base_models.py
"""

import os, pickle, time, json, warnings
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

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

# ─── Metric Functions ───

def ndcg_at_k(y_true, y_score, k):
    order = np.argsort(y_score)[::-1][:k]
    gains = y_true[order].astype(float)
    if gains.sum() == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = (gains / discounts).sum()
    ideal = np.sort(y_true.astype(float))[::-1][:k]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(y_true, y_score, k):
    top_k = np.argsort(y_score)[::-1][:k]
    return y_true[top_k].sum() / k


def recall_at_k(y_true, y_score, k):
    top_k = np.argsort(y_score)[::-1][:k]
    return y_true[top_k].sum() / max(y_true.sum(), 1)


def evaluate_per_group(df, score_col="score", group_col="session_id"):
    results = {"ndcg@3": [], "ndcg@5": [], "ndcg@10": [],
               "p@3": [], "p@5": [], "p@10": [],
               "r@3": [], "r@5": [], "r@10": []}
    for _, grp in df.groupby(group_col):
        yt = grp["was_accepted"].values
        ys = grp[score_col].values
        if yt.sum() == 0 or len(yt) < 2:
            continue
        for k in [3, 5, 10]:
            results[f"ndcg@{k}"].append(ndcg_at_k(yt, ys, k))
            results[f"p@{k}"].append(precision_at_k(yt, ys, k))
            results[f"r@{k}"].append(recall_at_k(yt, ys, k))
    return {k: round(np.mean(v), 4) if v else 0 for k, v in results.items()}


def print_metrics(name, metrics, auc=None):
    print(f"\n  [{name}]")
    if auc is not None:
        print(f"    AUC:         {auc:.4f}")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v:.4f}")


# ═══════════════════════════════════════════
# BASE MODEL 1: LightGBM Ranker
# ═══════════════════════════════════════════

def train_lightgbm(train_df, val_df, test_df, feature_cols):
    print("\n" + "=" * 55)
    print("  BASE MODEL 1: LightGBM LambdaRank")
    print("=" * 55)

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["was_accepted"].values
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df["was_accepted"].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["was_accepted"].values

    train_groups = train_df.groupby("session_id").size().values
    val_groups = val_df.groupby("session_id").size().values
    test_groups = test_df.groupby("session_id").size().values

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Optuna tuning
    N_TRIALS = 15
    print(f"  Running Optuna ({N_TRIALS} trials)...")

    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "verbose": -1,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }

        lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups, free_raw_data=False)
        lgb_val = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=lgb_train, free_raw_data=False)

        model = lgb.train(
            params, lgb_train,
            num_boost_round=300,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_val)
        val_copy = val_df.copy()
        val_copy["score"] = preds
        metrics = evaluate_per_group(val_copy, "score")
        return metrics.get("ndcg@5", 0)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_trial.params
    print(f"  Best NDCG@5: {study.best_value:.4f}")

    # Train final model
    final_params = {
        "objective": "lambdarank", "metric": "ndcg",
        "ndcg_eval_at": [3, 5, 10], "verbose": -1, "n_jobs": -1,
        **best_params,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train, group=train_groups, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=lgb_train, free_raw_data=False)

    t0 = time.time()
    model = lgb.train(
        final_params, lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s ({model.best_iteration} rounds)")

    # Evaluate on test
    test_preds = model.predict(X_test)
    test_copy = test_df.copy()
    test_copy["score"] = test_preds

    try:
        auc = roc_auc_score(y_test, test_preds)
    except:
        auc = 0.5
    metrics = evaluate_per_group(test_copy, "score")
    print_metrics("LightGBM", metrics, auc)

    # Feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)

    print("\n  Top-10 Features (gain):")
    for _, r in fi.head(10).iterrows():
        print(f"    {r['feature']:30s} gain={r['importance_gain']:.0f}")

    # Save
    with open(os.path.join(MODEL_DIR, "lgb_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    return model, metrics, auc, best_params


# ═══════════════════════════════════════════
# BASE MODEL 2: GRU Sequential Model
# ═══════════════════════════════════════════

def train_gru_model(train_df, val_df, test_df):
    print("\n" + "=" * 55)
    print("  BASE MODEL 2: GRU Sequential Ranker")
    print("=" * 55)

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("  PyTorch not installed. Skipping GRU model.")
        print("  Install with: pip install torch")
        # Save a dummy scorer
        with open(os.path.join(MODEL_DIR, "gru_model.pkl"), "wb") as f:
            pickle.dump({"type": "dummy"}, f)
        return None, {}, 0.5

    # Build item vocabulary
    items_df = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))
    item_ids = items_df["item_id"].unique().tolist()
    item2idx = {str(iid): idx + 1 for idx, iid in enumerate(item_ids)}  # 0 = padding
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

    class CartDataset(Dataset):
        def __init__(self, df, item2idx, max_seq_len=10):
            self.samples = []
            for _, row in df.iterrows():
                try:
                    cart = json.loads(row["cart_state_at_recommendation"]) if pd.notna(row["cart_state_at_recommendation"]) else []
                except:
                    cart = []

                cart_idx = [item2idx.get(str(c), 0) for c in cart]
                if len(cart_idx) == 0:
                    cart_idx = [0]
                cart_idx = cart_idx[-max_seq_len:]
                while len(cart_idx) < max_seq_len:
                    cart_idx.insert(0, 0)

                cand_idx = item2idx.get(str(row["recommended_item_id"]), 0)
                label = float(row["was_accepted"])
                self.samples.append((cart_idx, cand_idx, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            cart, cand, label = self.samples[idx]
            return (torch.tensor(cart, dtype=torch.long),
                    torch.tensor(cand, dtype=torch.long),
                    torch.tensor(label, dtype=torch.float32))

    import json

    print(f"  Vocab size: {vocab_size}")
    print("  Building datasets...")

    train_ds = CartDataset(train_df, item2idx)
    val_ds = CartDataset(val_df, item2idx)
    test_ds = CartDataset(test_df, item2idx)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    test_loader = DataLoader(test_ds, batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUCartEncoder(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    print(f"  Training on {device}...")
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(15):
        model.train()
        total_loss = 0
        for cart, cand, label in train_loader:
            cart, cand, label = cart.to(device), cand.to(device), label.to(device)
            pred = model(cart, cand)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for cart, cand, label in val_loader:
                cart, cand, label = cart.to(device), cand.to(device), label.to(device)
                pred = model(cart, cand)
                val_loss += criterion(pred, label).item()

        avg_train = total_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"    Epoch {epoch+1:2d}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gru_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # Evaluate on test
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gru_model.pt"), weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for cart, cand, label in test_loader:
            cart, cand = cart.to(device), cand.to(device)
            pred = model(cart, cand)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.numpy())

    test_copy = test_df.copy()
    test_copy["score"] = all_preds[:len(test_copy)]

    try:
        auc = roc_auc_score(all_labels[:len(test_copy)], all_preds[:len(test_copy)])
    except:
        auc = 0.5
    metrics = evaluate_per_group(test_copy, "score")
    print_metrics("GRU", metrics, auc)

    # Save item2idx mapping
    with open(os.path.join(MODEL_DIR, "gru_item2idx.pkl"), "wb") as f:
        pickle.dump(item2idx, f)

    return model, metrics, auc


# ═══════════════════════════════════════════
# BASE MODEL 3: Item-Item CF Scorer
# ═══════════════════════════════════════════

def train_cf_scorer(train_df, test_df):
    print("\n" + "=" * 55)
    print("  BASE MODEL 3: Item-Item CF Scorer")
    print("=" * 55)

    # Build co-occurrence matrix from order_items
    order_items = pd.read_csv(os.path.join(DATA_DIR, "order_items.csv"))
    order_groups = order_items.groupby("order_id")["item_id"].apply(list)

    co_matrix = {}
    for items in order_groups:
        for i in items:
            for j in items:
                if i != j:
                    co_matrix[(i, j)] = co_matrix.get((i, j), 0) + 1

    # Normalize
    max_count = max(co_matrix.values()) if co_matrix else 1
    co_matrix_norm = {k: v / max_count for k, v in co_matrix.items()}

    print(f"  Co-occurrence pairs: {len(co_matrix_norm)}")

    with open(os.path.join(MODEL_DIR, "co_occurrence_matrix.pkl"), "wb") as f:
        pickle.dump(co_matrix_norm, f)

    # Score test set
    def cf_score(cart_items_str, candidate_id):
        try:
            cart = json.loads(cart_items_str) if isinstance(cart_items_str, str) else []
        except:
            cart = []
        if not cart:
            return 0.0
        scores = [co_matrix_norm.get((c, candidate_id), 0) for c in cart]
        return float(np.mean(scores)) if scores else 0.0

    import json

    test_copy = test_df.copy()
    test_copy["score"] = test_copy.apply(
        lambda row: cf_score(row.get("cart_state_at_recommendation", "[]"), row["recommended_item_id"]),
        axis=1
    )

    try:
        auc = roc_auc_score(test_copy["was_accepted"], test_copy["score"])
    except:
        auc = 0.5
    metrics = evaluate_per_group(test_copy, "score")
    print_metrics("Item-Item CF", metrics, auc)

    return co_matrix_norm, metrics, auc


# ═══════════════════════════════════════════
# BASELINES
# ═══════════════════════════════════════════

def run_baselines(train_df, test_df):
    print("\n" + "=" * 55)
    print("  BASELINES")
    print("=" * 55)

    baselines = {}

    # Random
    print("\n  --- Random Ranker ---")
    test_copy = test_df.copy()
    test_copy["score"] = np.random.random(len(test_copy))
    try:
        auc = roc_auc_score(test_copy["was_accepted"], test_copy["score"])
    except:
        auc = 0.5
    metrics = evaluate_per_group(test_copy, "score")
    print_metrics("Random", metrics, auc)
    baselines["random"] = {**metrics, "auc": round(auc, 4)}

    # Popularity
    print("\n  --- Popularity Ranker ---")
    item_pop = train_df[train_df["was_accepted"] == 1].groupby("recommended_item_id").size()
    test_copy = test_df.copy()
    test_copy["score"] = test_copy["recommended_item_id"].map(item_pop).fillna(0)
    try:
        auc = roc_auc_score(test_copy["was_accepted"], test_copy["score"])
    except:
        auc = 0.5
    metrics = evaluate_per_group(test_copy, "score")
    print_metrics("Popularity", metrics, auc)
    baselines["popularity"] = {**metrics, "auc": round(auc, 4)}

    return baselines


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    print("=" * 60)
    print("  CSAO BASE MODEL TRAINING")
    print("=" * 60)

    print("Loading feature data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_features.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    print(f"  Features available: {len(feature_cols)}/{len(FEATURE_COLS)}")

    # Baselines
    baselines = run_baselines(train_df, test_df)

    # Base Model 1: LightGBM
    lgb_model, lgb_metrics, lgb_auc, lgb_params = train_lightgbm(
        train_df, val_df, test_df, feature_cols
    )

    # Base Model 2: GRU
    gru_model, gru_metrics, gru_auc = train_gru_model(train_df, val_df, test_df)

    # Base Model 3: CF
    cf_matrix, cf_metrics, cf_auc = train_cf_scorer(train_df, test_df)

    # Summary
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)

    all_results = {
        "baselines": baselines,
        "lgb": {**lgb_metrics, "auc": round(lgb_auc, 4), "params": lgb_params},
        "gru": {**gru_metrics, "auc": round(gru_auc, 4)},
        "cf": {**cf_metrics, "auc": round(cf_auc, 4)},
    }

    header = f"  {'Model':20s} {'AUC':>8s} {'NDCG@5':>8s} {'NDCG@10':>8s} {'P@5':>8s}"
    print(header)
    print("  " + "-" * len(header))
    for name, m in [("Random", baselines.get("random", {})),
                    ("Popularity", baselines.get("popularity", {})),
                    ("LightGBM", {**lgb_metrics, "auc": lgb_auc}),
                    ("GRU", {**gru_metrics, "auc": gru_auc}),
                    ("CF", {**cf_metrics, "auc": cf_auc})]:
        print(f"  {name:20s} {m.get('auc', 0):>8.4f} {m.get('ndcg@5', 0):>8.4f} "
              f"{m.get('ndcg@10', 0):>8.4f} {m.get('p@5', 0):>8.4f}")

    with open(os.path.join(MODEL_DIR, "training_metadata.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll models saved to {MODEL_DIR}/")
    print("Training complete!")


if __name__ == "__main__":
    main()
