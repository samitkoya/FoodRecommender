"""
cold_start_pipeline.py — Cold Start Handling for CSAO

Separate pipelines for each cold-start scenario:
  1. New User (<3 orders): city-level + time-slot popularity fallback
  2. New Restaurant: cuisine-level co-occurrence from similar restaurants
  3. New Item: embedding-based nearest neighbor inheritance

Run: python cold_start_pipeline.py
"""

import os, pickle, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

COLD_START_USER_THRESHOLD = 3
COLD_START_ITEM_THRESHOLD = 50


def build_city_popularity(orders_df, items_df):
    """Build city-level item popularity for new user fallback."""
    print("  Building city-level popularity...")
    merged = orders_df.merge(
        pd.read_csv(os.path.join(DATA_DIR, "order_items.csv")),
        on="order_id", how="inner"
    )
    city_pop = merged.groupby(["city", "item_id"]).size().reset_index(name="count")
    city_pop["rank"] = city_pop.groupby("city")["count"].rank(method="dense", ascending=False)
    return city_pop


def build_mealtime_popularity(interactions_df):
    """Build meal-time-slot popularity for time-aware fallback."""
    print("  Building meal-time popularity...")
    if "meal_time_slot" not in interactions_df.columns:
        return pd.DataFrame()

    accepted = interactions_df[interactions_df["was_accepted"] == 1]
    mt_pop = accepted.groupby(["meal_time_slot", "recommended_item_id"]).size().reset_index(name="count")
    mt_pop["rank"] = mt_pop.groupby("meal_time_slot")["count"].rank(method="dense", ascending=False)
    return mt_pop


def build_cuisine_co_occurrence(items_df, co_matrix):
    """Build cuisine-level co-occurrence for new restaurant fallback."""
    print("  Building cuisine-level co-occurrence...")
    item_cuisine = items_df.set_index("item_id")["cuisine_type"].to_dict()

    cuisine_co = {}
    for (a, b), score in co_matrix.items():
        ca = item_cuisine.get(a, "")
        cb = item_cuisine.get(b, "")
        if ca and cb:
            cuisine_co[(ca, cb)] = cuisine_co.get((ca, cb), 0) + score

    return cuisine_co


def recommend_for_new_user(user_city, meal_time, restaurant_id, cart_items,
                           city_pop, mt_pop, items_df, top_n=10):
    """New User (<3 orders): city + time-slot popularity fallback."""
    rest_items = items_df[items_df["restaurant_id"] == restaurant_id]["item_id"].tolist()
    candidates = [i for i in rest_items if i not in cart_items]

    if not candidates:
        return []

    # Score by city popularity
    city_scores = city_pop[city_pop["city"] == user_city].set_index("item_id")["count"].to_dict()

    # Score by mealtime popularity
    mt_scores = {}
    if not mt_pop.empty and meal_time:
        mt_sub = mt_pop[mt_pop["meal_time_slot"] == meal_time]
        mt_scores = mt_sub.set_index("recommended_item_id")["count"].to_dict()

    scored = []
    for cid in candidates:
        score = city_scores.get(cid, 0) * 0.6 + mt_scores.get(cid, 0) * 0.4
        scored.append((cid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def recommend_for_new_restaurant(cuisine_type, cart_items, items_df, cuisine_co, top_n=10):
    """New Restaurant: cuisine-level co-occurrence from similar restaurants."""
    # Find similar restaurants by cuisine
    similar_items = items_df[items_df["cuisine_type"] == cuisine_type]["item_id"].tolist()
    candidates = [i for i in similar_items if i not in cart_items]

    if not candidates:
        return []

    scored = []
    for cid in candidates:
        score = cuisine_co.get((cuisine_type, cuisine_type), 0)
        scored.append((cid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def recommend_for_new_item(item_id, items_df, co_matrix, top_n=5):
    """New Item: find K nearest items by category+price and inherit their stats."""
    if item_id not in items_df.index:
        return []

    item = items_df.loc[item_id]
    cat = item.get("category", "")
    price = item.get("price", 0)

    # Find similar items by category and price
    same_cat = items_df[items_df["category"] == cat]
    same_cat = same_cat.copy()
    same_cat["price_diff"] = abs(same_cat["price"] - price)
    nearest = same_cat.nsmallest(top_n + 1, "price_diff")
    nearest = nearest[nearest.index != item_id]

    result = []
    for nid in nearest.index[:top_n]:
        # Inherit co-occurrence stats
        inherited_score = sum(v for (a, b), v in co_matrix.items() if a == nid or b == nid)
        result.append((nid, inherited_score))

    return result


def main():
    print("=" * 55)
    print("  CSAO COLD START PIPELINE")
    print("=" * 55)

    items_df = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))
    orders_df = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    users_df = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))

    # Load co-occurrence matrix
    co_path = os.path.join(MODEL_DIR, "co_occurrence_matrix.pkl")
    co_matrix = {}
    if os.path.exists(co_path):
        with open(co_path, "rb") as f:
            co_matrix = pickle.load(f)

    # Load training data for interaction-based stats
    train_path = os.path.join(DATA_DIR, "csao_training_data.csv")
    if os.path.exists(train_path):
        interactions = pd.read_csv(train_path)
    else:
        interactions = pd.DataFrame()

    # Build lookup tables
    city_pop = build_city_popularity(orders_df, items_df)
    mt_pop = build_mealtime_popularity(interactions)
    cuisine_co = build_cuisine_co_occurrence(items_df, co_matrix)

    # Save cold-start artifacts
    cold_start_data = {
        "city_popularity": city_pop.to_dict("records") if not city_pop.empty else [],
        "cuisine_co_occurrence_pairs": len(cuisine_co),
    }

    with open(os.path.join(MODEL_DIR, "cold_start_data.pkl"), "wb") as f:
        pickle.dump({
            "city_pop": city_pop,
            "mt_pop": mt_pop,
            "cuisine_co": cuisine_co,
        }, f)

    # Stats
    cold_users = users_df[users_df["total_orders"] < COLD_START_USER_THRESHOLD]
    print(f"\n  Cold-start users: {len(cold_users):,} ({len(cold_users)/len(users_df)*100:.1f}%)")
    print(f"  Co-occurrence matrix size: {len(co_matrix):,}")
    print(f"  Cuisine co-occurrence pairs: {len(cuisine_co):,}")
    print(f"  City popularity entries: {len(city_pop):,}")

    # Demo
    print("\n  --- Demo: New User Recommendation ---")
    demo_recs = recommend_for_new_user(
        "Mumbai", "dinner", items_df["restaurant_id"].iloc[0], [],
        city_pop, mt_pop, items_df, top_n=5
    )
    for iid, score in demo_recs[:5]:
        name = items_df[items_df["item_id"] == iid]["item_name"].values
        name = name[0] if len(name) > 0 else "?"
        print(f"    {name:30s} score={score:.2f}")

    print("\nCold start pipeline complete!")


if __name__ == "__main__":
    main()
