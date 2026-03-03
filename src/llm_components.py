"""
llm_components.py — LLM-Powered Components for CSAO

A. Semantic Item Embeddings (sentence-transformers, local, free)
   - Embeds item names/descriptions into dense vectors
   - For cold-start items: find K nearest neighbors in embedding space

B. Meal Coherence Scoring (Gemini API scaffold, offline batch)
   - Pre-compute meal completeness scores for common cart patterns
   - NOT called in real-time inference

C. LLM-Based Menu Understanding (Gemini API scaffold, offline)
   - Bootstrap item-item affinity for new restaurants

Run: python llm_components.py
"""

import os, pickle, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ═══════════════════════════════════════════
# A. Semantic Item Embeddings (Free, Local)
# ═══════════════════════════════════════════

def compute_item_embeddings(items_df):
    """
    Use sentence-transformers to embed item names into dense vectors.
    Model: all-MiniLM-L6-v2 (~80MB, runs on CPU, no API needed).
    """
    print("\n--- A. Semantic Item Embeddings ---")

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  Model loaded: all-MiniLM-L6-v2")

        texts = []
        item_ids = []
        for _, row in items_df.iterrows():
            name = str(row.get("item_name", ""))
            cat = str(row.get("category", ""))
            cuisine = str(row.get("cuisine_type", ""))
            text = f"{name} - {cat} - {cuisine}"
            texts.append(text)
            item_ids.append(row["item_id"])

        print(f"  Encoding {len(texts)} items...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

        # Save embeddings
        emb_dict = {iid: emb for iid, emb in zip(item_ids, embeddings)}
        with open(os.path.join(MODEL_DIR, "item_embeddings.pkl"), "wb") as f:
            pickle.dump(emb_dict, f)

        print(f"  Saved {len(emb_dict)} embeddings (dim={embeddings.shape[1]})")
        return emb_dict

    except ImportError:
        print("  sentence-transformers not installed.")
        print("  Install: pip install sentence-transformers")
        print("  Generating random embeddings as placeholder...")

        emb_dict = {}
        for _, row in items_df.iterrows():
            emb_dict[row["item_id"]] = np.random.randn(384).astype(np.float32)

        with open(os.path.join(MODEL_DIR, "item_embeddings.pkl"), "wb") as f:
            pickle.dump(emb_dict, f)
        print(f"  Saved {len(emb_dict)} placeholder embeddings (dim=384)")
        return emb_dict


def find_similar_items(item_id, emb_dict, k=5):
    """Find K nearest items by cosine similarity in embedding space."""
    if item_id not in emb_dict:
        return []

    target = emb_dict[item_id]
    scores = []
    for iid, emb in emb_dict.items():
        if iid == item_id:
            continue
        sim = np.dot(target, emb) / (np.linalg.norm(target) * np.linalg.norm(emb) + 1e-8)
        scores.append((iid, float(sim)))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


# ═══════════════════════════════════════════
# B. Meal Coherence Scoring (Offline Batch)
# ═══════════════════════════════════════════

def compute_meal_coherence_scores():
    """
    Pre-compute meal coherence scores for common cart patterns.
    Uses Gemini API free tier (offline batch, NOT real-time).

    To use: Get free API key from https://aistudio.google.com
    Set environment variable: GEMINI_API_KEY=your_key
    """
    print("\n--- B. Meal Coherence Scoring ---")

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        print("  GEMINI_API_KEY not set. Using rule-based fallback.")
        print("  To enable: export GEMINI_API_KEY=your_free_api_key")

        # Rule-based fallback
        common_carts = [
            (["Biryani"], 0.3, "Side", "Add Raita or Salan"),
            (["Biryani", "Raita"], 0.5, "Beverage", "Add a cold drink"),
            (["Biryani", "Raita", "Coke"], 0.7, "Dessert", "Add Gulab Jamun"),
            (["Butter Chicken"], 0.3, "Side", "Add Naan or Rice"),
            (["Butter Chicken", "Garlic Naan"], 0.6, "Beverage", "Add Lassi"),
            (["Pizza"], 0.25, "Beverage", "Add Coke or Sprite"),
            (["Pizza", "Coke"], 0.5, "Side", "Add Fries or Garlic Bread"),
            (["Dosa"], 0.3, "Beverage", "Add Masala Chai"),
        ]

        results = []
        for cart, score, missing, suggestion in common_carts:
            results.append({
                "cart_items": cart,
                "completeness_score": score,
                "missing_category": missing,
                "suggested_item_type": suggestion,
            })

        with open(os.path.join(MODEL_DIR, "meal_coherence_cache.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {len(results)} rule-based coherence scores")
        return results

    # Gemini API path
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel("gemini-2.0-flash")

        items_df = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))

        # Top-20 most common cart patterns (simplified)
        common_patterns = [
            ["Biryani"],
            ["Biryani", "Raita"],
            ["Butter Chicken", "Naan"],
            ["Pizza"],
            ["Dosa"],
        ]

        results = []
        for cart in common_patterns:
            prompt = f"""You are a food expert. Given cart items: {cart}
Rate 0-1 how complete this meal feels. What category is missing?
Output JSON: {{"completeness_score": float, "missing_category": str, "suggested_item_type": str}}"""

            try:
                response = gemini.generate_content(prompt)
                text = response.text.strip()
                if text.startswith("```"): text = text.split("```")[1].strip("json\n")
                data = json.loads(text)
                data["cart_items"] = cart
                results.append(data)
                print(f"  Cart {cart}: completeness={data.get('completeness_score', 0)}")
            except Exception as e:
                print(f"  Cart {cart}: API error - {e}")

        with open(os.path.join(MODEL_DIR, "meal_coherence_cache.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {len(results)} LLM coherence scores")
        return results

    except ImportError:
        print("  google-generativeai not installed.")
        print("  Install: pip install google-generativeai")
        return []


# ═══════════════════════════════════════════
# C. Menu Understanding for New Restaurants
# ═══════════════════════════════════════════

def bootstrap_new_restaurant_affinity():
    """
    Use LLM to infer item-item affinity from menu text.
    Offline job — runs once when a new restaurant is onboarded.
    """
    print("\n--- C. Menu Understanding for New Restaurants ---")

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        print("  GEMINI_API_KEY not set. Using category-based rules.")

        # Rule-based fallback
        affinity_rules = {
            "Main": ["Side", "Beverage", "Starter"],
            "Side": ["Main", "Beverage"],
            "Beverage": ["Main", "Dessert"],
            "Dessert": ["Main", "Beverage"],
            "Starter": ["Main", "Beverage"],
        }

        with open(os.path.join(MODEL_DIR, "restaurant_affinity_rules.json"), "w") as f:
            json.dump(affinity_rules, f, indent=2)
        print(f"  Saved category-based affinity rules")
        return affinity_rules

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel("gemini-2.0-flash")

        items_df = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))
        sample_rest = items_df[items_df["restaurant_id"] == items_df["restaurant_id"].iloc[0]]
        menu_text = ", ".join(sample_rest["item_name"].tolist())

        prompt = f"""Given this restaurant menu: {menu_text}
For each item, suggest top 3 complementary add-ons from the same menu.
Format: {{"item_name": ["complement_1", "complement_2", "complement_3"]}}"""

        response = gemini.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"): text = text.split("```")[1].strip("json\n")
        affinities = json.loads(text)

        with open(os.path.join(MODEL_DIR, "restaurant_affinity_llm.json"), "w") as f:
            json.dump(affinities, f, indent=2)

        print(f"  Generated affinities for {len(affinities)} items")
        return affinities

    except Exception as e:
        print(f"  LLM menu understanding failed: {e}")
        return {}


def main():
    print("=" * 55)
    print("  CSAO LLM COMPONENTS")
    print("=" * 55)

    items_df = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))

    # A. Item embeddings
    emb_dict = compute_item_embeddings(items_df)

    # Demo: find similar items
    sample_id = items_df["item_id"].iloc[0]
    similar = find_similar_items(sample_id, emb_dict, k=3)
    print(f"\n  Similar to {sample_id}:")
    for sid, sim in similar:
        name = items_df[items_df["item_id"] == sid]["item_name"].values
        name = name[0] if len(name) > 0 else "?"
        print(f"    {name:30s} similarity={sim:.4f}")

    # B. Meal coherence
    compute_meal_coherence_scores()

    # C. Menu understanding
    bootstrap_new_restaurant_affinity()

    print("\nLLM components complete!")


if __name__ == "__main__":
    main()
