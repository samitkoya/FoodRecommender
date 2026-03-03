"""
ab_test_analysis.py — A/B Testing Framework for CSAO

Designs and analyzes online experiments:
  - Power analysis and sample size calculation
  - Two-proportion z-test for acceptance rate
  - T-test for AOV lift
  - Mann-Whitney U for non-normal distributions
  - Bonferroni correction for multiple metrics
  - Guardrail metrics monitoring
  - Segment-level analysis

Run: python ab_test_analysis.py
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def power_analysis(baseline_rate=0.15, mde=0.02, alpha=0.05, power=0.80):
    """Calculate minimum sample size per arm."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_avg = (p1 + p2) / 2

    n = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg))
         + z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2 / mde**2

    return int(np.ceil(n))


def two_proportion_ztest(n_control, successes_control, n_treatment, successes_treatment):
    """Two-proportion z-test for acceptance rate comparison."""
    p1 = successes_control / n_control
    p2 = successes_treatment / n_treatment
    p_pool = (successes_control + successes_treatment) / (n_control + n_treatment)

    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
    z = (p2 - p1) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    ci_low = (p2 - p1) - 1.96 * se
    ci_high = (p2 - p1) + 1.96 * se

    return {
        "z_statistic": round(z, 4),
        "p_value": round(p_value, 6),
        "control_rate": round(p1, 4),
        "treatment_rate": round(p2, 4),
        "lift": round((p2 - p1) / p1 * 100, 2) if p1 > 0 else 0,
        "ci_95": (round(ci_low, 4), round(ci_high, 4)),
        "significant": p_value < 0.05,
    }


def ttest_aov_lift(control_aov, treatment_aov):
    """T-test for average order value lift."""
    t_stat, p_value = stats.ttest_ind(treatment_aov, control_aov)
    cohens_d = (np.mean(treatment_aov) - np.mean(control_aov)) / np.sqrt(
        (np.std(control_aov)**2 + np.std(treatment_aov)**2) / 2
    )

    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "control_mean_aov": round(np.mean(control_aov), 2),
        "treatment_mean_aov": round(np.mean(treatment_aov), 2),
        "aov_lift": round(np.mean(treatment_aov) - np.mean(control_aov), 2),
        "cohens_d": round(cohens_d, 4),
        "significant": p_value < 0.05,
    }


def mannwhitney_test(control, treatment):
    """Mann-Whitney U test for non-normal distributions."""
    u_stat, p_value = stats.mannwhitneyu(treatment, control, alternative="two-sided")
    return {
        "u_statistic": round(u_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
    }


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple comparisons."""
    n = len(p_values)
    adjusted = {k: min(v * n, 1.0) for k, v in p_values.items()}
    significant = {k: v < alpha for k, v in adjusted.items()}
    return adjusted, significant


def check_guardrails(control_metrics, treatment_metrics):
    """Check guardrail metrics (must NOT degrade)."""
    guardrails = {}

    # Cart-to-order ratio
    c2o_control = control_metrics.get("c2o_ratio", 0.85)
    c2o_treatment = treatment_metrics.get("c2o_ratio", 0.85)
    c2o_drop = c2o_control - c2o_treatment
    guardrails["c2o_ratio"] = {
        "control": c2o_control,
        "treatment": c2o_treatment,
        "degradation": round(c2o_drop, 4),
        "passed": c2o_drop < 0.005,  # Must not drop > 0.5%
    }

    # Order completion rate
    ocr_control = control_metrics.get("order_completion_rate", 0.92)
    ocr_treatment = treatment_metrics.get("order_completion_rate", 0.92)
    ocr_drop = ocr_control - ocr_treatment
    guardrails["order_completion"] = {
        "control": ocr_control,
        "treatment": ocr_treatment,
        "degradation": round(ocr_drop, 4),
        "passed": ocr_drop < 0.01,
    }

    return guardrails


def simulate_and_analyze():
    """Run a full simulated A/B test analysis."""
    np.random.seed(42)

    n_per_arm = 50000

    # Simulate control (baseline popularity-based)
    control_accepts = np.random.binomial(1, 0.15, n_per_arm)
    control_aov = np.random.normal(350, 100, n_per_arm)

    # Simulate treatment (ML model — higher acceptance)
    treatment_accepts = np.random.binomial(1, 0.22, n_per_arm)
    treatment_aov = np.random.normal(380, 110, n_per_arm)

    return control_accepts, treatment_accepts, control_aov, treatment_aov, n_per_arm


def main():
    print("=" * 55)
    print("  CSAO A/B TEST ANALYSIS")
    print("=" * 55)

    # 1. Experiment Design
    print("\n--- Experiment Design ---")
    print("  Randomization: USER_ID (not session)")
    print("  Traffic split: 50/50")
    print("  Duration: 2 weeks minimum")

    n_per_arm = power_analysis(baseline_rate=0.15, mde=0.02)
    print(f"\n  Power Analysis:")
    print(f"    Baseline CTR: 15%")
    print(f"    MDE: 2%")
    print(f"    Alpha: 0.05, Power: 0.80")
    print(f"    Required sample per arm: {n_per_arm:,}")
    print(f"    Total: {n_per_arm * 2:,}")

    # 2. Run simulated analysis
    control_acc, treatment_acc, control_aov, treatment_aov, n = simulate_and_analyze()

    # 3. Primary Metrics
    print("\n--- Primary Metrics ---")

    # Acceptance Rate
    accept_result = two_proportion_ztest(
        n, control_acc.sum(), n, treatment_acc.sum()
    )
    print(f"\n  Acceptance Rate:")
    print(f"    Control:   {accept_result['control_rate']:.4f}")
    print(f"    Treatment: {accept_result['treatment_rate']:.4f}")
    print(f"    Lift:      {accept_result['lift']:.2f}%")
    print(f"    p-value:   {accept_result['p_value']:.6f}")
    print(f"    95% CI:    {accept_result['ci_95']}")
    print(f"    Significant: {'YES' if accept_result['significant'] else 'NO'}")

    # AOV Lift
    aov_result = ttest_aov_lift(control_aov, treatment_aov)
    print(f"\n  AOV Lift:")
    print(f"    Control mean:   ₹{aov_result['control_mean_aov']:.2f}")
    print(f"    Treatment mean: ₹{aov_result['treatment_mean_aov']:.2f}")
    print(f"    Lift:           ₹{aov_result['aov_lift']:.2f}")
    print(f"    Cohen's d:      {aov_result['cohens_d']:.4f}")
    print(f"    p-value:        {aov_result['p_value']:.6f}")
    print(f"    Significant: {'YES' if aov_result['significant'] else 'NO'}")

    # 4. Bonferroni correction
    print("\n--- Multiple Testing Correction ---")
    p_values = {"acceptance_rate": accept_result["p_value"], "aov_lift": aov_result["p_value"]}
    adjusted, sig = bonferroni_correction(p_values)
    for metric, adj_p in adjusted.items():
        print(f"  {metric}: adjusted p={adj_p:.6f}, significant={sig[metric]}")

    # 5. Guardrails
    print("\n--- Guardrail Metrics ---")
    guardrails = check_guardrails(
        {"c2o_ratio": 0.85, "order_completion_rate": 0.92},
        {"c2o_ratio": 0.848, "order_completion_rate": 0.919},
    )
    for name, g in guardrails.items():
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  {name}: control={g['control']}, treatment={g['treatment']}, "
              f"degradation={g['degradation']:.4f} [{status}]")

    # 6. Business Impact
    print("\n--- Business Impact Projection ---")
    daily_sessions = 5_000_000
    csao_pct = 0.60
    avg_addon = 80
    sessions = daily_sessions * csao_pct

    current = sessions * 3 * accept_result["control_rate"] * avg_addon
    projected = sessions * 3 * accept_result["treatment_rate"] * avg_addon
    lift = projected - current

    print(f"  Daily sessions with CSAO: {sessions:,.0f}")
    print(f"  Current daily revenue:    ₹{current:,.0f}")
    print(f"  Projected revenue:        ₹{projected:,.0f}")
    print(f"  Daily lift:               ₹{lift:,.0f}")
    print(f"  Annual lift:              ₹{lift * 365:,.0f}")

    print("\nA/B test analysis complete!")


if __name__ == "__main__":
    main()
