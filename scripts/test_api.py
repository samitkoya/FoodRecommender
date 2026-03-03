"""
test_api.py — API Test Suite for CSAO Recommendation Server

Tests:
  1. Health check
  2. Single cart recommendation
  3. Multi-item cart
  4. Sequential cart (add item → get recs → add recommended → get new recs)
  5. Cold-start user
  6. Latency stress test (50 requests)

Run AFTER starting the server:
  uvicorn inference_service:app --host 0.0.0.0 --port 8000
  python scripts/test_api.py
"""

import requests, json, time, sys
import numpy as np

BASE = "http://localhost:8000"


def test_health():
    print("[1] Health Check")
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        data = r.json()
        print(f"    Status: {data['status']}")
        print(f"    Model: {data['model']}")
        print(f"    Users: {data['users_loaded']}, Items: {data['items_loaded']}")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        print("    Start server: uvicorn inference_service:app --port 8000")
        return False


def test_single_cart():
    print("\n[2] Single Cart Recommendation")
    payload = {
        "user_id": "1",
        "restaurant_id": "1",
        "cart_items": ["2"],
        "n_recommendations": 8,
    }
    r = requests.post(f"{BASE}/v1/csao/recommend", json=payload)
    data = r.json()
    print(f"    Latency: {data['latency_ms']}ms")
    print(f"    Ensemble path: {data.get('ensemble_path', 'N/A')}")
    print(f"    Cold start: {data.get('is_cold_start', 'N/A')}")
    print(f"    Top Recommendations:")
    for rec in data["recommendations"][:5]:
        print(f"      [{rec['category']:10s}] {rec['item_name']:30s} ₹{rec['price']:.0f}  score={rec['score']}")
    return True


def test_multi_item_cart():
    print("\n[3] Multi-Item Cart")
    payload = {
        "user_id": "42",
        "restaurant_id": "5",
        "cart_items": ["100", "101"],
        "n_recommendations": 5,
    }
    r = requests.post(f"{BASE}/v1/csao/recommend", json=payload)
    data = r.json()
    print(f"    Cart: 2 items, Latency: {data['latency_ms']}ms")
    for rec in data["recommendations"]:
        print(f"      [{rec['category']:10s}] {rec['item_name']:30s} score={rec['score']}")
    return True


def test_sequential_cart():
    print("\n[4] Sequential Cart Test")
    uid, rid = "10", "3"

    p1 = {"user_id": uid, "restaurant_id": rid, "cart_items": ["50"], "n_recommendations": 5}
    r1 = requests.post(f"{BASE}/v1/csao/recommend", json=p1)
    d1 = r1.json()
    print(f"    Step 1 - Cart: [i_50], Latency: {d1['latency_ms']}ms")
    for rec in d1["recommendations"][:3]:
        print(f"      -> {rec['item_name']:30s} score={rec['score']}")

    if d1["recommendations"]:
        added = d1["recommendations"][0]["item_id"]
        p2 = {"user_id": uid, "restaurant_id": rid, "cart_items": ["50", added], "n_recommendations": 5}
        r2 = requests.post(f"{BASE}/v1/csao/recommend", json=p2)
        d2 = r2.json()
        print(f"\n    Step 2 - Added '{d1['recommendations'][0]['item_name']}', Latency: {d2['latency_ms']}ms")
        for rec in d2["recommendations"][:3]:
            print(f"      -> {rec['item_name']:30s} score={rec['score']}")

        ids1 = {r["item_id"] for r in d1["recommendations"]}
        ids2 = {r["item_id"] for r in d2["recommendations"]}
        print(f"\n    Recommendations changed: {'YES' if ids1 != ids2 else 'NO'}")
    return True


def test_cold_start():
    print("\n[5] Cold-Start User (unknown user)")
    payload = {
        "user_id": "new_user_999",
        "restaurant_id": "10",
        "cart_items": ["200"],
        "n_recommendations": 5,
    }
    r = requests.post(f"{BASE}/v1/csao/recommend", json=payload)
    data = r.json()
    print(f"    Cold start: {data.get('is_cold_start', 'N/A')}")
    print(f"    Latency: {data['latency_ms']}ms")
    print(f"    Returns {len(data['recommendations'])} recommendations (popularity fallback)")
    return True


def test_latency_stress():
    print("\n[6] Latency Stress Test (50 requests)...")
    latencies = []
    failures = 0
    for i in range(1, 51):
        payload = {
            "user_id": f"u_{i}",
            "restaurant_id": f"r_{(i % 10) + 1}",
            "cart_items": [f"i_{(i % 50) + 1}"],
            "n_recommendations": 8,
        }
        t0 = time.time()
        try:
            resp = requests.post(f"{BASE}/v1/csao/recommend", json=payload)
            latencies.append((time.time() - t0) * 1000)
        except:
            failures += 1

    if latencies:
        print(f"    p50: {np.percentile(latencies, 50):.1f}ms")
        print(f"    p90: {np.percentile(latencies, 90):.1f}ms")
        print(f"    p99: {np.percentile(latencies, 99):.1f}ms")
        print(f"    Max: {max(latencies):.1f}ms")
        under_200 = sum(1 for l in latencies if l < 200)
        print(f"    {under_200}/{len(latencies)} under 200ms SLA")
        if failures:
            print(f"    {failures} requests failed")
    return True


def main():
    print("=" * 55)
    print("  CSAO API TEST SUITE")
    print("=" * 55)

    if not test_health():
        print("\nServer not reachable. Aborting.")
        sys.exit(1)

    test_single_cart()
    test_multi_item_cart()
    test_sequential_cart()
    test_cold_start()
    test_latency_stress()

    print("\n" + "=" * 55)
    print("  ALL TESTS COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()
