"""
feature_pipeline.py — Feature Engineering for CSAO Recommendation System

Reads ONLY from csao_training_data.csv (the merged flat file).
Computes all feature groups: meal completion, user, cart context,
candidate item, contextual, co-occurrence, and cold-start features.

Run: python feature_pipeline.py
"""

import os
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Encoding Maps ──
SEG_MAP = {"budget": 0, "regular": 1, "premium": 2}
PRICE_MAP = {"budget": 0, "mid": 1, "premium": 2}
CAT_MAP = {"Main": 0, "Side": 1, "Beverage": 2, "Dessert": 3, "Starter": 4}
MEAL_MAP = {"breakfast": 0, "lunch": 1, "evening_snack": 2, "dinner": 3, "late_night": 4}


def safe_json_loads(val):
    if pd.isna(val):
        return []
    try:
        return json.loads(val)
    except:
        return []


def compute_meal_completion_features(df, items_lookup):
    """Group A: Meal Completion Features — the primary differentiator."""
    print("  Computing meal completion features...")

    meal_has_main = []
    meal_has_side = []
    meal_has_beverage = []
    meal_has_dessert = []
    meal_completeness = []
    candidate_fills = []
    cuisine_coherence = []

    for _, row in df.iterrows():
        cart_items = safe_json_loads(row.get("cart_state_at_recommendation", "[]"))
        candidate_cat = row.get("category", "")

        cart_cats = set()
        cart_cuisines = []
        for cid in cart_items:
            info = items_lookup.get(str(cid), {})
            cart_cats.add(info.get("category", ""))
            cart_cuisines.append(info.get("cuisine_type", ""))

        has_main = int("Main" in cart_cats)
        has_side = int("Side" in cart_cats)
        has_bev = int("Beverage" in cart_cats)
        has_des = int("Dessert" in cart_cats)

        meal_has_main.append(has_main)
        meal_has_side.append(has_side)
        meal_has_beverage.append(has_bev)
        meal_has_dessert.append(has_des)

        present = sum([has_main, has_side, has_bev, has_des])
        meal_completeness.append(present / 4.0)

        missing_cats = []
        if not has_main: missing_cats.append("Main")
        if not has_side: missing_cats.append("Side")
        if not has_bev: missing_cats.append("Beverage")
        if not has_des: missing_cats.append("Dessert")
        candidate_fills.append(int(candidate_cat in missing_cats))

        if len(cart_cuisines) > 0:
            cand_cuisine = row.get("cuisine_type", "")
            matches = sum(1 for c in cart_cuisines if c == cand_cuisine)
            cuisine_coherence.append(matches / len(cart_cuisines))
        else:
            cuisine_coherence.append(0.0)

    df["meal_has_main"] = meal_has_main
    df["meal_has_side"] = meal_has_side
    df["meal_has_beverage"] = meal_has_beverage
    df["meal_has_dessert"] = meal_has_dessert
    df["meal_completeness_score"] = meal_completeness
    df["candidate_fills_gap"] = candidate_fills
    df["cuisine_coherence_score"] = cuisine_coherence
    return df


def compute_user_features(df):
    """Group B: User Features."""
    print("  Computing user features...")
    df["user_segment_enc"] = df["user_segment"].map(SEG_MAP).fillna(1)
    df["user_order_frequency"] = df["total_orders"].fillna(0)
    df["user_avg_order_value"] = df["avg_order_value"].fillna(300)
    df["is_cold_start_user"] = (df["total_orders"].fillna(0) < 3).astype(int)

    # Compute per-user addon acceptance rate from the dataset itself
    user_accept = df.groupby("user_id")["was_accepted"].mean().rename("user_addon_acceptance_rate")
    df = df.merge(user_accept, on="user_id", how="left")
    df["user_addon_acceptance_rate"] = df["user_addon_acceptance_rate"].fillna(0.15)

    # Price sensitivity: ratio of avg_order_value to segment median
    seg_median = df.groupby("user_segment")["avg_order_value"].transform("median")
    df["user_price_sensitivity"] = (df["avg_order_value"] / (seg_median + 1)).fillna(1.0)

    df["days_since_last_order"] = np.random.randint(0, 30, size=len(df))  # simulated
    return df


def compute_cart_context_features(df, items_lookup):
    """Group C: Cart Context Features."""
    print("  Computing cart context features...")

    cart_total = []
    cart_count = []
    cart_avg_price = []
    cart_single = []
    cand_pct = []

    for _, row in df.iterrows():
        cart_items = safe_json_loads(row.get("cart_state_at_recommendation", "[]"))
        prices = [items_lookup.get(str(cid), {}).get("price", 0) for cid in cart_items]
        total = sum(prices)
        count = len(cart_items)
        avg_p = total / max(count, 1)

        cart_total.append(total)
        cart_count.append(count)
        cart_avg_price.append(avg_p)
        cart_single.append(int(count == 1))

        cand_price = row.get("price", 0)
        if pd.isna(cand_price): cand_price = 0
        cand_pct.append(cand_price / max(total, 1))

    df["cart_total_value"] = cart_total
    df["cart_item_count"] = cart_count
    df["cart_avg_item_price"] = cart_avg_price
    df["cart_is_single_item"] = cart_single
    df["candidate_price_pct_of_cart"] = cand_pct
    return df


def compute_candidate_features(df):
    """Group D: Candidate Item Features."""
    print("  Computing candidate item features...")
    df["item_category_enc"] = df["category"].map(CAT_MAP).fillna(0)
    df["item_price"] = df["price"].fillna(df["price"].median() if "price" in df.columns else 200)
    df["item_is_veg"] = df["is_veg"].fillna(0).astype(int)
    df["item_avg_rating"] = df["avg_rating"].fillna(4.0)
    df["is_cold_start_item"] = 0  # default, overridden by cold_start_pipeline

    # Popularity rank within restaurant
    df["item_popularity_rank"] = df.groupby("restaurant_id")["is_popular"].rank(
        method="dense", ascending=False
    ).fillna(1).astype(int)

    # Attachment rate: how often this item is accepted when recommended
    if "item_attachment_rate" in df.columns:
        df = df.drop(columns=["item_attachment_rate"])
    item_attach = df.groupby("recommended_item_id")["was_accepted"].mean().rename("item_attachment_rate")
    df = df.merge(item_attach, left_on="recommended_item_id", right_index=True, how="left")
    df["item_attachment_rate"] = df["item_attachment_rate"].fillna(0.1)
    return df


def compute_contextual_features(df):
    """Group E: Contextual Features."""
    print("  Computing contextual features...")
    df["meal_time_enc"] = df["meal_time_slot"].map(MEAL_MAP).fillna(3)
    # hour_of_day, day_of_week, is_weekend already computed in build_training_dataset
    return df


def compute_co_occurrence_features(df):
    """Group F: Co-occurrence and Sequential Features."""
    print("  Computing co-occurrence features...")
    # Normalize raw scores
    max_co = df["co_occurrence_score_raw"].max()
    if max_co > 0:
        df["co_occurrence_score"] = df["co_occurrence_score_raw"] / max_co
    else:
        df["co_occurrence_score"] = 0.0

    max_seq = df["sequential_transition_score_raw"].max()
    if max_seq > 0:
        df["sequential_transition_score"] = df["sequential_transition_score_raw"] / max_seq
    else:
        df["sequential_transition_score"] = 0.0
    return df


def main():
    print("=" * 55)
    print("  CSAO FEATURE ENGINEERING PIPELINE")
    print("=" * 55)

    input_path = os.path.join(DATA_DIR, "csao_training_data.csv")
    print(f"Reading from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Shape: {df.shape}")

    # Build items lookup for cart-based features
    items_df = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))
    items_lookup = {}
    for _, row in items_df.iterrows():
        items_lookup[str(row["item_id"])] = row.to_dict()

    df = compute_meal_completion_features(df, items_lookup)
    df = compute_user_features(df)
    df = compute_cart_context_features(df, items_lookup)
    df = compute_candidate_features(df)
    df = compute_contextual_features(df)
    df = compute_co_occurrence_features(df)

    # ── Temporal Split ──
    print("  Applying temporal split...")
    df["recommendation_timestamp"] = pd.to_datetime(df["recommendation_timestamp"])
    df = df.sort_values("recommendation_timestamp")

    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_df.to_csv(os.path.join(DATA_DIR, "train_features.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val_features.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_features.csv"), index=False)

    print(f"\n  Train: {train_df.shape}")
    print(f"  Val:   {val_df.shape}")
    print(f"  Test:  {test_df.shape}")
    print(f"  Positive rate (train): {train_df['was_accepted'].mean():.3f}")
    print(f"  Positive rate (test):  {test_df['was_accepted'].mean():.3f}")
    print("\nFeature pipeline complete!")


if __name__ == "__main__":
    main()
