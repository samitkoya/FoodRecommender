"""
ab_test_analysis.py — Streamlined A/B Testing Analysis for CSAO

ponytail: Simplified redundant stat-harness code.
Uses scipy.stats directly for two-proportion Z-test and T-test for AOV.
"""

import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def two_proportion_ztest(n_control, succ_c, n_treatment, succ_t):
    """Two-proportion z-test for acceptance rate comparison."""
    p1, p2 = succ_c / n_control, succ_t / n_treatment
    p_pool = (succ_c + succ_t) / (n_control + n_treatment)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = (p2 - p1) / se if se > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"control_rate": round(p1, 4), "treatment_rate": round(p2, 4), "lift_pct": round((p2 - p1) / p1 * 100, 2), "p_value": round(p_val, 6)}


def ttest_aov_lift(control_aov, treatment_aov):
    """T-test for average order value lift."""
    t_stat, p_val = stats.ttest_ind(treatment_aov, control_aov)
    return {"control_mean": round(np.mean(control_aov), 2), "treatment_mean": round(np.mean(treatment_aov), 2), "lift": round(np.mean(treatment_aov) - np.mean(control_aov), 2), "p_value": round(p_val, 6)}


def main():
    print("=" * 55)
    print("  CSAO A/B TEST ANALYSIS (STREAMLINED)")
    print("=" * 55)

    np.random.seed(42)
    n = 50000

    c_acc = np.random.binomial(1, 0.15, n)
    t_acc = np.random.binomial(1, 0.22, n)
    c_aov = np.random.normal(350, 100, n)
    t_aov = np.random.normal(380, 110, n)

    res_acc = two_proportion_ztest(n, c_acc.sum(), n, t_acc.sum())
    res_aov = ttest_aov_lift(c_aov, t_aov)

    print(f"\n  Acceptance Rate: Control={res_acc['control_rate']}, Treatment={res_acc['treatment_rate']} (Lift: {res_acc['lift_pct']}%, p={res_acc['p_value']})")
    print(f"  AOV Lift:        Control=₹{res_aov['control_mean']}, Treatment=₹{res_aov['treatment_mean']} (Lift: ₹{res_aov['lift']}, p={res_aov['p_value']})")

    daily_sessions = 3_000_000
    daily_lift = daily_sessions * 3 * (res_acc["treatment_rate"] - res_acc["control_rate"]) * 80
    print(f"\n  Projected Daily Lift: ₹{daily_lift:,.0f} | Annual Lift: ₹{daily_lift * 365:,.0f}")
    print("\nA/B test analysis complete!")


if __name__ == "__main__":
    main()
