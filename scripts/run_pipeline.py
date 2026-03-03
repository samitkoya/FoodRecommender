"""
run_pipeline.py — End-to-End CSAO Pipeline Orchestrator

Runs all steps in sequence:
  1. Generate synthetic data (7 normalized CSVs)
  2. Build flat training dataset
  3. Feature engineering
  4. Train 3 base models (LightGBM, GRU, CF)
  5. Train meta-learner
  6. Evaluate full model
  7. Cold start pipeline
  8. A/B test analysis

Run: python scripts/run_pipeline.py
"""

import sys, os, time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)
os.chdir(BASE_DIR)


def run_step(name, func):
    print(f"\n{'#' * 60}")
    print(f"  STEP: {name}")
    print(f"{'#' * 60}")
    t0 = time.time()
    func()
    elapsed = time.time() - t0
    print(f"\n  [{name}] completed in {elapsed:.1f}s")
    return elapsed


def main():
    print("=" * 60)
    print("  CSAO RECOMMENDATION SYSTEM — FULL PIPELINE")
    print("=" * 60)

    timings = {}

    # 1. Generate synthetic data
    from generate_synthetic_data import main as gen_data
    timings["Data Generation"] = run_step("Data Generation", gen_data)

    # 2. Build training dataset
    from build_training_dataset import main as build_data
    timings["Build Training Data"] = run_step("Build Training Data", build_data)

    # 3. Feature engineering
    from feature_pipeline import main as feat_eng
    timings["Feature Engineering"] = run_step("Feature Engineering", feat_eng)

    # 4. Train base models
    from train_base_models import main as train_models
    timings["Train Base Models"] = run_step("Train Base Models", train_models)

    # 5. Train meta-learner
    from train_meta_learner import main as train_meta
    timings["Train Meta-Learner"] = run_step("Train Meta-Learner", train_meta)

    # 6. Evaluate
    from evaluate_model import main as evaluate
    timings["Evaluation"] = run_step("Evaluation", evaluate)

    # 7. Cold start
    from cold_start_pipeline import main as cold_start
    timings["Cold Start"] = run_step("Cold Start Pipeline", cold_start)

    # 8. A/B test analysis
    from ab_test_analysis import main as ab_test
    timings["A/B Test Analysis"] = run_step("A/B Test Analysis", ab_test)

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    total = sum(timings.values())
    for step, t in timings.items():
        print(f"  {step:25s}  {t:8.1f}s")
    print(f"  {'TOTAL':25s}  {total:8.1f}s")

    print("\nNext steps:")
    print("  1. Start API:     uvicorn src.inference_service:app --host 0.0.0.0 --port 8000")
    print("  2. Test API:      python scripts/test_api.py")
    print("  3. API docs:      http://localhost:8000/docs")
    print("  4. Dashboard:     Open ui/index.html in your browser")
    print("  5. LLM features:  python src/llm_components.py")
    print("  6. Charts:        reports/evaluation_charts.png")


if __name__ == "__main__":
    main()
