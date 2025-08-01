import numpy as np
import statsmodels.stats.multitest as smm
from scipy import stats

# --- Holm-Bonferroni Correction ---


def holmbonferroni_correction(raw_p_values_dict):
    print("--- Holm-Bonferroni Correction using statsmodels ---")

    # Extract p-values and labels for statsmodels
    labels = list(raw_p_values_dict.keys())
    p_values = np.array(list(raw_p_values_dict.values()))

    alpha = 0.05

    # Apply Holm-Bonferroni correction
    reject, pvals_corrected, _, _ = smm.multipletests(
        p_values, alpha=alpha, method="holm"
    )

    print(f"Original p-values:\n{p_values}\n")
    print(f"Labels:\n{labels}\n")
    print(f"Alpha level: {alpha}\n")

    print("--- Detailed Holm-Bonferroni Results ---")
    for i in range(len(labels)):
        print(f"Comparison: {labels[i]}")
        print(f"  Raw p-value: {p_values[i]:.4f}")
        print(f"  Holm-adjusted p-value: {pvals_corrected[i]:.4f}")
        print(f"  Significant (after correction): {reject[i]}")
        print("-" * 30)


# --- Function for Obtaining Effect Sizes (Cohen's d_z) ---


def calculate_cohens_dz_with_ci(scores_condition_a, scores_condition_b, alpha=0.05):
    """
    Calculates Cohen's d_z for paired samples, with standard error and confidence interval.

    Args:
        scores_condition_a (array-like): Scores from condition A.
        scores_condition_b (array-like): Scores from condition B.
        alpha (float): Significance level for confidence interval (default=0.05 for 95%).

    Returns:
        dict: {
            'd_z': float,
            'se': float,
            'ci_low': float,
            'ci_high': float
        }
    """
    print("\n--- Calculating Effect Size (Cohen's d_z) with Confidence Interval ---")

    a = np.asarray(scores_condition_a)
    b = np.asarray(scores_condition_b)

    if len(a) != len(b):
        raise ValueError("Input arrays must have the same length.")

    n = len(a)
    diff = a - b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)  # sample standard deviation

    if std_diff == 0:
        return {"d_z": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    d_z = mean_diff / std_diff

    # Standard error of d_z (Lakens, 2013)
    se = (1 / np.sqrt(n)) * np.sqrt(1 + (d_z**2 / 2))

    # t critical value
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    ci_low = d_z - t_crit * se
    ci_high = d_z + t_crit * se

    return {"d_z": d_z, "se": se, "ci_low": ci_low, "ci_high": ci_high}


def interpret_cohens_dz(cohens_dz):
    """
    Provides a qualitative interpretation of Cohen's d_z.

    Args:
        cohens_dz (float): Cohen's d_z value.

    Returns:
        str: Interpretation (e.g., "Small effect").
    """
    abs_dz = abs(cohens_dz)
    if abs_dz < 0.2:
        return "Very small effect"
    elif abs_dz < 0.5:
        return "Small effect"
    elif abs_dz < 0.8:
        return "Medium effect"
    else:
        return "Large effect"
