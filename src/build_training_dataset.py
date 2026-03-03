import os
import pandas as pd
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def build_co_occurrence_matrix(order_items):
    print("Building co-occurrence matrix...")
    order_groups = order_items.groupby("order_id")["item_id"].apply(list)
    
    co_occur = {}
    for items in order_groups:
        for i in items:
            for j in items:
                if i != j:
                    pair = (i, j)
                    co_occur[pair] = co_occur.get(pair, 0) + 1
                    
    return co_occur

def build_sequential_matrix(sessions):
    print("Building sequential transition matrix...")
    seq_matrix = {}
    for seq_str in sessions["items_added_sequence"].dropna():
        seq = json.loads(seq_str)
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            seq_matrix[pair] = seq_matrix.get(pair, 0) + 1
    return seq_matrix

def main():
    print("Loading raw tables...")
    csao = pd.read_csv(os.path.join(DATA_DIR, "csao_interactions.csv"))
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    rests = pd.read_csv(os.path.join(DATA_DIR, "restaurants.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))
    sessions = pd.read_csv(os.path.join(DATA_DIR, "cart_sessions.csv"))
    order_items = pd.read_csv(os.path.join(DATA_DIR, "order_items.csv"))

    co_occur_matrix = build_co_occurrence_matrix(order_items)
    seq_matrix = build_sequential_matrix(sessions)

    print("Joining tables...")
    df = csao.merge(users, on="user_id", how="left")
    df = df.merge(sessions[["session_id", "session_datetime", "items_added_sequence", "final_cart_items", "did_order"]], on="session_id", how="left")
    df = df.merge(rests, on="restaurant_id", how="left", suffixes=("", "_rest"))
    
    df.rename(columns={"city": "user_city", "name": "restaurant_name"}, inplace=True)
    
    df = df.merge(items, left_on="recommended_item_id", right_on="item_id", how="left", suffixes=("", "_candidate"))
    
    df["recommendation_timestamp"] = pd.to_datetime(df["recommendation_timestamp"])
    df["hour_of_day"] = df["recommendation_timestamp"].dt.hour
    df["day_of_week"] = df["recommendation_timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    def get_meal_time(hour):
        if 6 <= hour < 11: return "breakfast"
        elif 11 <= hour < 15: return "lunch"
        elif 15 <= hour < 19: return "evening_snack"
        elif 19 <= hour < 23: return "dinner"
        return "late_night"

    df["meal_time_slot"] = df["hour_of_day"].apply(get_meal_time)

    # Compute co-occurrence and sequential scores and append to flattened table
    print("Pre-computing co-occurrence scores...")
    co_scores = []
    seq_scores = []
    for _, row in df.iterrows():
        try:
            cart = json.loads(row["cart_state_at_recommendation"])
        except:
            cart = []
            
        candidate = row["recommended_item_id"]
        
        c_score = 0
        if len(cart) > 0:
            for item in cart:
                c_score += co_occur_matrix.get((item, candidate), 0)
            c_score /= len(cart)
        co_scores.append(c_score)
        
        s_score = 0
        if len(cart) > 0:
            last_item = cart[-1]
            s_score = seq_matrix.get((last_item, candidate), 0)
        seq_scores.append(s_score)
        
    df["co_occurrence_score_raw"] = co_scores
    df["sequential_transition_score_raw"] = seq_scores

    out_path = os.path.join(DATA_DIR, "csao_training_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Flattening complete! Saved to {out_path} with shape {df.shape}")

if __name__ == "__main__":
    main()
