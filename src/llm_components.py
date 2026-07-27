"""
llm_components.py — Lightweight Item Components & Rules for CSAO

ponytail: Removed sentence-transformers and Google GenAI dependencies.
Uses deterministic category/price heuristics and item feature vectors.
Ceiling: Static rule matrix. Upgrade path: Re-introduce vector search if high-dim semantic retrieval is required.
"""

import os, json, pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def compute_item_embeddings(items_df):
    """Generate dense feature-based embeddings for cold-start items."""
    print("\n--- A. Item Feature Embeddings ---")
    emb_dict = {}
    
    cat_map = {"Main": [1, 0, 0, 0], "Side": [0, 1, 0, 0], "Beverage": [0, 0, 1, 0], "Dessert": [0, 0, 0, 1]}
    
    for _, row in items_df.iterrows():
        cat_vec = cat_map.get(row.get("category", "Main"), [0.25, 0.25, 0.25, 0.25])
        price_norm = float(row.get("price", 100)) / 500.0
        is_veg = float(row.get("is_veg", 1))
        
        # 6-dim deterministic feature vector
        vec = np.array(cat_vec + [price_norm, is_veg], dtype=np.float32)
        emb_dict[row["item_id"]] = vec

    with open(os.path.join(MODEL_DIR, "item_embeddings.pkl"), "wb") as f:
        pickle.dump(emb_dict, f)

    print(f"  Saved {len(emb_dict)} feature embeddings (dim=6)")
    return emb_dict


def find_similar_items(item_id, emb_dict, k=5):
    """Find K nearest items by cosine similarity."""
    if item_id not in emb_dict:
        return []

    target = emb_dict[item_id]
    target_norm = np.linalg.norm(target) + 1e-8
    
    scores = []
    for iid, emb in emb_dict.items():
        if iid == item_id:
            continue
        sim = np.dot(target, emb) / ((np.linalg.norm(emb) + 1e-8) * target_norm)
        scores.append((iid, float(sim)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def main():
    print("=" * 55)
    print("  CSAO LIGHTWEIGHT COMPONENTS")
    print("=" * 55)

    menu_path = os.path.join(DATA_DIR, "menu_items.csv")
    if os.path.exists(menu_path):
        items_df = pd.read_csv(menu_path)
        emb_dict = compute_item_embeddings(items_df)
        sample_id = items_df["item_id"].iloc[0]
        similar = find_similar_items(sample_id, emb_dict, k=3)
        print(f"\n  Similar to {sample_id}: {similar[:2]}")

    print("\nLightweight components complete!")


if __name__ == "__main__":
    main()

