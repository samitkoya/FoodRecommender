"""
inference_service.py — FastAPI Inference Server for CSAO Recommendations

Production-hardened API with:
  - Stacked ensemble (LightGBM + GRU + CF + Meta-Learner)
  - In-memory dict lookups for fast feature building
  - Sequential cart update logic
  - Cold-start fallback (popularity-based for unknown users)
  - Latency tracking
  - Diversity filter for recommendations

Run: uvicorn inference_service:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""

import os, sys, pickle, json, time, warnings
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure sibling modules (ensemble_inference) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, conlist
from typing import List, Optional
from google import genai
from dotenv import load_dotenv

load_dotenv()
try:
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
except Exception as e:
    print("Warning: Gemini client not initialized", e)
    gemini_client = None

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
UI_DIR = os.path.join(BASE_DIR, "ui")

SEG_MAP = {"budget": 0, "regular": 1, "premium": 2}
PRICE_MAP = {"budget": 0, "mid": 1, "premium": 2}
CAT_MAP = {"Main": 0, "Side": 1, "Beverage": 2, "Dessert": 3, "Starter": 4}
MEAL_MAP = {"breakfast": 0, "lunch": 1, "evening_snack": 2, "dinner": 3, "late_night": 4}

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

# ─── Load Data ───
print("Loading data for inference...")

USERS_DF = pd.read_csv(os.path.join(DATA_DIR, "users.csv"), dtype={"user_id": str}).set_index("user_id")
ITEMS_DF = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"), dtype={"item_id": str, "restaurant_id": str}).set_index("item_id")
RESTS_DF = pd.read_csv(os.path.join(DATA_DIR, "restaurants.csv"), dtype={"restaurant_id": str}).set_index("restaurant_id")

# Load orders for history
try:
    ORDERS_DF = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"), dtype={"order_id": str, "user_id": str})
    ORDER_ITEMS_DF = pd.read_csv(os.path.join(DATA_DIR, "order_items.csv"), dtype={"order_id": str, "item_id": str})
except FileNotFoundError:
    ORDERS_DF = pd.DataFrame(columns=["order_id", "user_id", "restaurant_id", "order_time"])
    ORDER_ITEMS_DF = pd.DataFrame(columns=["order_id", "item_id", "quantity", "price"])

USERS_DICT = USERS_DF.to_dict("index")
ITEMS_DICT = ITEMS_DF.to_dict("index")
RESTS_DICT = RESTS_DF.to_dict("index")

REST_ITEMS = {}
for item_id, row in ITEMS_DF.iterrows():
    rid = row["restaurant_id"]
    REST_ITEMS.setdefault(rid, []).append(item_id)

CO_LOOKUP = {}
co_path = os.path.join(MODEL_DIR, "co_occurrence_matrix.pkl")
if os.path.exists(co_path):
    with open(co_path, "rb") as f:
        CO_LOOKUP = pickle.load(f)

ITEM_POPULARITY = {}
if "is_popular" in ITEMS_DF.columns:
    for iid, row in ITEMS_DF.iterrows():
        ITEM_POPULARITY[iid] = 1.0 if row.get("is_popular", False) else 0.5

# Load ensemble ranker
from ensemble_inference import EnsembleRanker
RANKER = EnsembleRanker()
RANKER.load_models()

meta_path = os.path.join(MODEL_DIR, "training_metadata.json")
TRAINING_META = {}
if os.path.exists(meta_path):
    with open(meta_path) as f:
        TRAINING_META = json.load(f)

print(f"  {len(USERS_DICT)} users, {len(RESTS_DICT)} restaurants, {len(ITEMS_DICT)} items")
print(f"  {len(CO_LOOKUP)} co-occurrence pairs")
print("Server ready!")

# ─── FastAPI App ───
app = FastAPI(
    title="CSAO Recommendation API",
    version="2.0",
    description="Cart Super Add-On Rail — Stacked Ensemble Recommendations"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ───
class RecommendRequest(BaseModel):
    user_id: str
    restaurant_id: str
    cart_items: List[str]
    session_id: Optional[str] = None
    context: Optional[dict] = None
    n_recommendations: int = 10


class RecommendationItem(BaseModel):
    item_id: str
    item_name: str
    category: str
    price: float
    score: float
    rank: int
    reason: str
    is_veg: bool
    tagline: str = ""


class RecommendResponse(BaseModel):
    recommendations: List[RecommendationItem]
    latency_ms: float
    model_version: str = "v2.1.0"
    is_cold_start: bool
    ensemble_path: str


# ─── Helpers ───
def get_meal_time_enc(hour):
    if 6 <= hour < 11: return 0
    elif 11 <= hour < 15: return 1
    elif 15 <= hour < 19: return 2
    elif 19 <= hour < 23: return 3
    return 4


def determine_reason(candidate_cat, cart_cats):
    if candidate_cat not in cart_cats:
        return candidate_cat.lower()
    return "complementary"


def build_request_features(req, hour):
    uid = req.user_id
    rid = req.restaurant_id
    is_cold_start = uid not in USERS_DICT
    user = USERS_DICT.get(uid, next(iter(USERS_DICT.values())))
    rest = RESTS_DICT.get(rid, next(iter(RESTS_DICT.values())))

    # Use a set of STRING ids for fast exclusion; keep full list for quantity-aware metrics
    cart_item_list  = req.cart_items                              # full list (may have dupes for qty > 1)
    cart_ids_set    = set(str(x) for x in cart_item_list)        # unique string ids
    cart_n          = len(cart_item_list)                         # total quantity
    cart_value      = sum(float(ITEMS_DICT.get(str(cid), {}).get("price", 0)) for cid in cart_item_list)

    cart_cats = set()
    for cid in cart_ids_set:
        cart_cats.add(ITEMS_DICT.get(str(cid), {}).get("category", ""))

    has_main = int("Main" in cart_cats)
    has_side = int("Side" in cart_cats)
    has_bev  = int("Beverage" in cart_cats)
    has_des  = int("Dessert" in cart_cats)
    completeness = sum([has_main, has_side, has_bev, has_des]) / 4.0
    missing = [c for c in ["Main", "Side", "Beverage", "Dessert"] if c not in cart_cats]

    # Get the restaurant cuisine for coherence filtering
    rest_cuisine = str(rest.get("cuisine_type", "")).strip().lower()

    rest_item_ids = REST_ITEMS.get(rid, [])
    # Exclude cart items (string comparison) and cross-cuisine items to prevent
    # e.g. Chinese food appearing in a Biryani restaurant's recommendations
    filtered_ids = []
    for iid in rest_item_ids:
        if str(iid) in cart_ids_set:
            continue
        item_data = ITEMS_DICT.get(iid, {})
        item_cuisine = str(item_data.get("cuisine_type", "")).strip().lower()
        # Keep item if either side has no cuisine tag, OR cuisines match
        if not rest_cuisine or not item_cuisine or item_cuisine == rest_cuisine:
            filtered_ids.append(iid)
    candidate_ids = filtered_ids

    if not candidate_ids:
        return pd.DataFrame(), [], is_cold_start, {}

    meal_enc = get_meal_time_enc(hour)
    dow = datetime.now().weekday()

    rows = []
    for iid in candidate_ids:
        item = ITEMS_DICT.get(iid, {})
        co_score = 0.0
        if cart_ids_set:
            co_score = sum(CO_LOOKUP.get((c, iid), 0) for c in cart_ids_set) / (cart_n + 1)

        cat = item.get("category", "Main")
        fills_gap = int(cat in missing)

        rows.append({
            "user_segment_enc": SEG_MAP.get(user.get("user_segment", "regular"), 1),
            "user_order_frequency": user.get("total_orders", 10),
            "user_avg_order_value": user.get("avg_order_value", 300),
            "user_addon_acceptance_rate": 0.15,
            "user_price_sensitivity": 1.0,
            "days_since_last_order": 5,
            "is_cold_start_user": int(is_cold_start),
            "cart_total_value": cart_value,
            "cart_item_count": cart_n,
            "cart_avg_item_price": cart_value / max(cart_n, 1),
            "cart_is_single_item": int(cart_n == 1),
            "candidate_price_pct_of_cart": item.get("price", 200) / max(cart_value, 1),
            "meal_has_main": has_main,
            "meal_has_side": has_side,
            "meal_has_beverage": has_bev,
            "meal_has_dessert": has_des,
            "meal_completeness_score": completeness,
            "candidate_fills_gap": fills_gap,
            "cuisine_coherence_score": 1.0 if str(item.get("cuisine_type", "")).lower() == rest_cuisine else 0.5,
            "item_category_enc": CAT_MAP.get(cat, 0),
            "item_price": item.get("price", 200),
            "item_is_veg": int(item.get("is_veg", False)),
            "item_popularity_rank": 1 if item.get("is_popular", False) else 5,
            "item_avg_rating": item.get("avg_rating", 4.0),
            "item_attachment_rate": 0.1,
            "co_occurrence_score": co_score,
            "hour_of_day": hour,
            "day_of_week": dow,
            "meal_time_enc": meal_enc,
            "is_weekend": int(dow >= 5),
            "is_cold_start_item": 0,
            "sequential_transition_score": 0.0,
        })

    feat_df = pd.DataFrame(rows)
    context = {
        "user_segment_enc": SEG_MAP.get(user.get("user_segment", "regular"), 1),
        "meal_time_enc": meal_enc,
        "is_cold_start_user": int(is_cold_start),
        "cart_item_count": cart_n,
        "meal_completeness_score": completeness,
    }

    return feat_df, candidate_ids, is_cold_start, context


# ─── Routes ───
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "CSAO Stacked Ensemble v2.1",
        "users_loaded": len(USERS_DICT),
        "items_loaded": len(ITEMS_DICT),
        "restaurants_loaded": len(RESTS_DICT),
        "co_occurrence_pairs": len(CO_LOOKUP),
    }

@app.get("/v1/restaurants")
def get_restaurants():
    """Return ALL restaurants with city field so the frontend can filter by city."""
    rest_list = []
    for rid, row in RESTS_DICT.items():
        rest_list.append({
            "id":            str(rid),
            "name":          row.get("name", "Unknown"),
            "city":          row.get("city", ""),          # ← was missing before
            "cuisine":       row.get("cuisine_type", ""),
            "price_range":   row.get("price_range", "mid"),
            "rating":        round(float(row.get("avg_rating", 4.0)), 1),
            "delivery_time": int(row.get("avg_delivery_time_min", 30)),
        })
    return {"restaurants": rest_list}

@app.get("/v1/restaurants/{restaurant_id}/menu")
def get_restaurant_menu(restaurant_id: str):
    """Return all menu items for a restaurant — used by the menu modal."""
    # REST_ITEMS maps restaurant_id -> [item_ids]
    items = REST_ITEMS.get(restaurant_id, [])

    # Fallback: try int key (IDs may be stored as int in the dict)
    if not items:
        try:
            items = REST_ITEMS.get(int(restaurant_id), [])
        except (ValueError, TypeError):
            pass

    if not items:
        raise HTTPException(status_code=404, detail=f"No menu found for restaurant {restaurant_id}")

    menu = []
    for item_id in items:
        item = ITEMS_DICT.get(str(item_id)) or ITEMS_DICT.get(item_id)
        if not item:
            continue
        menu.append({
            "item_id":    str(item_id),
            "item_name":  item.get("item_name", "Unknown Item"),
            "category":   item.get("category", "Main"),
            "price":      round(float(item.get("price", 0)), 2),
            "is_veg":     bool(item.get("is_veg", False)),
            "avg_rating": round(float(item.get("avg_rating", 4.0)), 1),
            "is_popular": bool(item.get("is_popular", False)),
        })

    # Sort by category display order
    cat_order = {"Starter": 0, "Main": 1, "Side": 2, "Beverage": 3, "Dessert": 4}
    menu.sort(key=lambda x: (cat_order.get(x["category"], 9), x["item_name"]))

    return {"restaurant_id": restaurant_id, "items": menu}


@app.get("/v1/user/active/history")
def get_active_user_history():
    """Return the most recent order for any active user."""
    if ORDERS_DF.empty:
        raise HTTPException(status_code=404, detail="No order history found.")

    # Sort by order_datetime descending to get the most recent order
    try:
        sorted_orders = ORDERS_DF.sort_values("order_datetime", ascending=False)
        last_order = sorted_orders.iloc[0]
    except Exception:
        last_order = ORDERS_DF.iloc[-1]

    order_id = str(last_order["order_id"])
    rest_id  = str(last_order["restaurant_id"])
    user_id  = str(last_order["user_id"])

    # Safe datetime extraction (pandas Series doesn't support .get())
    try:
        order_time = str(last_order["order_datetime"])
    except Exception:
        order_time = ""

    order_items_rows = ORDER_ITEMS_DF[ORDER_ITEMS_DF["order_id"] == order_id]

    items = []
    total = 0.0
    for _, item_row in order_items_rows.iterrows():
        iid          = str(item_row["item_id"])
        item_details = ITEMS_DICT.get(iid, {})
        # Use .get() on dict item_row but check column existence first
        price = float(item_row["price"] if ("price" in item_row and pd.notna(item_row["price"])) else item_details.get("price", 0))
        qty   = int(item_row["quantity"] if ("quantity" in item_row and pd.notna(item_row["quantity"])) else 1)
        items.append({
            "item_id":   iid,
            "item_name": item_details.get("item_name", "Unknown Item"),
            "category":  item_details.get("category", ""),
            "price":     price,
            "quantity":  qty,
        })
        total += price * qty

    rest_name = RESTS_DICT.get(rest_id, {}).get("name", "Unknown Restaurant")

    return {
        "user_id":         user_id,       # ← required for reorder to use correct user for AI recs
        "order_id":        order_id,
        "restaurant_id":   rest_id,
        "restaurant_name": rest_name,
        "order_time":      order_time,
        "items":           items,
        "total":           float(total),
    }

@app.get("/v1/user/{user_id}/history")
def get_user_history(user_id: str):
    """Return the most recent order for a given user."""
    user_orders = ORDERS_DF[ORDERS_DF["user_id"] == user_id]
    if user_orders.empty:
        raise HTTPException(status_code=404, detail="No order history found for this user.")

    # Sort by order_datetime descending
    try:
        user_orders = user_orders.sort_values("order_datetime", ascending=False)
    except Exception:
        pass

    last_order = user_orders.iloc[0]
    order_id = str(last_order["order_id"])
    rest_id  = str(last_order["restaurant_id"])

    try:
        order_time = str(last_order["order_datetime"])
    except Exception:
        order_time = ""

    order_items_rows = ORDER_ITEMS_DF[ORDER_ITEMS_DF["order_id"] == order_id]

    items = []
    total = 0.0
    for _, item_row in order_items_rows.iterrows():
        iid          = str(item_row["item_id"])
        item_details = ITEMS_DICT.get(iid, {})
        price = float(item_row["price"] if ("price" in item_row and pd.notna(item_row["price"])) else item_details.get("price", 0))
        qty   = int(item_row["quantity"] if ("quantity" in item_row and pd.notna(item_row["quantity"])) else 1)
        items.append({
            "item_id":   iid,
            "item_name": item_details.get("item_name", "Unknown Item"),
            "category":  item_details.get("category", ""),
            "price":     price,
            "quantity":  qty,
        })
        total += price * qty

    rest_name = RESTS_DICT.get(rest_id, {}).get("name", "Unknown Restaurant")

    return {
        "user_id":         user_id,
        "order_id":        order_id,
        "restaurant_id":   rest_id,
        "restaurant_name": rest_name,
        "order_time":      order_time,
        "items":           items,
        "total":           float(total),
    }

@app.get("/v1/order/track/{order_id}")
def track_order(order_id: str):
    """Simulate tracking an order based on elapsed time."""
    # Let's mock the start time for the demo as ~just now or use order history if available.
    # In a real app we would check ORDERS_DF. For this demo, we'll randomize or calculate based on order_id.
    
    # We will simulate the elapsed time using python hash of order_id to give stable-ish but changing results
    # Or actually simpler: just mock 4 states rotating based on current minute or user clicks
    import time
    fake_elapsed = int(time.time() * 10) % 1800  # cycles between 0 to 1800 seconds (30 mins)
    
    if fake_elapsed < 60:
        status = "Order Received"
        message = "Restaurant is confirming your order."
        step = 1
    elif fake_elapsed < 300:
        status = "Preparing Food"
        message = "Your food is being prepared."
        step = 2
    elif fake_elapsed < 1200:
        status = "Out for Delivery"
        message = "Your order is on the way."
        step = 3
    else:
        status = "Delivered"
        message = "Enjoy your meal!"
        step = 4
        
    return {
        "order_id": order_id,
        "status": status,
        "message": message,
        "step": step,
        "driver": "Rajan K. ⭐ 4.8",
        "vehicle": "🛵 MH-02-AB-1234",
        "eta_mins": max(1, 30 - (fake_elapsed // 60))
    }

@app.get("/")
def serve_ui():
    """Serves the frontend application."""
    return FileResponse(os.path.join(UI_DIR, "index.html"))


# ════════════════════════════════════════════════════════════════════════════
# 4. INFERENCE ENDPOINT
# ════════════════════════════════════════════════════════════════════════════
@app.post("/v1/csao/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t0 = time.time()
    hour = datetime.now().hour
    if req.context and "timestamp" in req.context:
        try:
            dt = datetime.fromisoformat(req.context["timestamp"].replace("Z", "+00:00"))
            hour = dt.hour
        except:
            pass

    feat_df, candidate_ids, is_cold_start, context = build_request_features(req, hour)

    if len(feat_df) == 0:
        raise HTTPException(status_code=404, detail="No candidate items found for this restaurant.")

    if is_cold_start:
        # Popularity fallback for cold-start users
        scores = np.array([ITEM_POPULARITY.get(iid, 0.5) for iid in candidate_ids])
        ensemble_path = "cold_start_popularity"
    else:
        scores, latency_info = RANKER.rank_sync(
            feat_df, list(req.cart_items), candidate_ids, context
        )
        ensemble_path = latency_info.get("path", "full_ensemble")

    # Diversity filter
    selected_idx = RANKER.apply_diversity_filter(
        scores, candidate_ids, ITEMS_DICT, top_n=req.n_recommendations
    )

    cart_cats = set()
    for cid in req.cart_items:
        cart_cats.add(ITEMS_DICT.get(cid, {}).get("category", ""))

    recs = []
    
    # Send chosen items to Gemini to generate taglines
    gemini_prompt = "You are an AI assistant for a food delivery app. A user has these items in their cart:\n"
    cart_item_names = [ITEMS_DICT.get(cid, {}).get("item_name", "Unknown") for cid in req.cart_items]
    gemini_prompt += ", ".join(cart_item_names) + "\n\n"
    gemini_prompt += "Here are some recommended add-ons:\n"
    
    candidate_details = []
    for rank, idx in enumerate(selected_idx, 1):
        iid = candidate_ids[idx]
        item = ITEMS_DICT.get(iid, {})
        candidate_details.append(f"{rank}. {item.get('item_name', 'Unknown')}")
    
    gemini_prompt += "\n".join(candidate_details) + "\n\n"
    gemini_prompt += "For each recommended item, write a short, catchy 1-sentence tagline explaining why it pairs perfectly with the cart.\n"
    gemini_prompt += "Return ONLY a JSON array of strings in the exact same order as the recommendations. No markdown formatting."
    
    taglines = []
    try:
        if os.environ.get("GEMINI_API_KEY") and gemini_client:
            resp = gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=gemini_prompt
            )
            # Try to parse JSON array from response
            text = resp.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.endswith("```"): text = text[:-3]
            taglines = json.loads(text.strip())
    except Exception as e:
        print(f"Gemini error: {e}")
        
    for rank, idx in enumerate(selected_idx, 1):
        iid = candidate_ids[idx]
        item = ITEMS_DICT.get(iid, {})
        cat = item.get("category", "Unknown")
        
        tagline = "Perfect addition to your meal!"
        if rank - 1 < len(taglines):
            tagline = taglines[rank - 1]
            
        recs.append(RecommendationItem(
            item_id=str(iid),
            item_name=str(item.get("item_name", "Unknown")),
            category=cat,
            price=float(item.get("price", 0)),
            score=round(float(scores[idx]), 4),
            rank=rank,
            reason=determine_reason(cat, cart_cats),
            is_veg=bool(item.get("is_veg", False)),
            tagline=tagline
        ))

    latency = (time.time() - t0) * 1000
    return RecommendResponse(
        recommendations=recs,
        latency_ms=round(latency, 2),
        is_cold_start=is_cold_start,
        ensemble_path=ensemble_path,
    )


@app.post("/recommend")
def recommend_legacy(req: RecommendRequest):
    """Legacy endpoint for backwards compatibility."""
    return recommend(req)


# Mount static files correctly
if os.path.exists(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")
    
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(UI_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_service:app", host="0.0.0.0", port=8000, reload=True)
