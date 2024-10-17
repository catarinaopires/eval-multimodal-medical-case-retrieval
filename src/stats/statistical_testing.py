import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from dotenv import load_dotenv

load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH")
OUTPUT_DIR_PATH = os.getenv("OUTPUT_DIR_PATH")


def read_map_scores(path):
    """
    Read MAP scores from a file.

    Parameters:
    path (str): Path to the file containing MAP scores

    Returns:
    map_scores (list): List of tuples containing topic numbers and MAP scores
    """
    map_scores = []

    with open(path, "r") as f:
        for line in f:
            _, topic_nr, map_score = line.split()
            if topic_nr != "all":
                map_scores.append((topic_nr, float(map_score)))

    map_scores.sort(key=lambda x: x[0])
    map_scores = [x[1] for x in map_scores]

    return map_scores


def parse_scores(output):
    scores = []

    for line in output.splitlines():
        _, topic_nr, score = line.split()
        if topic_nr != "all":
            scores.append((topic_nr, float(score)))

    scores.sort(key=lambda x: x[0])
    scores = [x[1] for x in scores]

    return scores


# -----------------------------------
# Normality Testing
# Perform Shapiro-Wilk test for normality
def perform_shapiro_wilk_test(data):
    """
    Perform a Shapiro-Wilk test for normality on a set of data.

    Parameters:
    data (array-like): Set of data

    Returns:
    stat (float): Test statistic
    p_value (float): P-value
    """
    stat, p_value = stats.shapiro(data)
    print(f"Run1 Shapiro-Wilk test p-value: {p_value}")

    return stat, p_value


def plot_qq_hist(data, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax[0])
    ax[0].set_title(f"Q-Q Plot: {title}")

    # Histogram with fitted normal distribution
    sns.histplot(data, kde=True, ax=ax[1], stat="density", linewidth=0)
    mean, std = np.mean(data), np.std(data)
    xmin, xmax = ax[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std)
    ax[1].plot(x, p, "k", linewidth=2)
    ax[1].set_title(f"Histogram: {title}")

    plt.tight_layout()
    # plt.show()

    plt.savefig(OUTPUT_DIR_PATH + f"stats/{title}_qq_hist.png")
    plt.close()


# -----------------------------------
# Statistical Testing
# Paired Sample T-Test
# WHEN: Both samples are normally distributed
def perform_paired_sample_t_test(data1, data2):
    """
    Perform a paired sample t-test on two sets of data.

    Parameters:
    data1 (array-like): First set of data
    data2 (array-like): Second set of data

    Returns:
    t_stat (float): T-statistic value
    p_value (float): P-value
    """
    try:
        t_stat, p_value = stats.ttest_rel(data1, data2)
    except ValueError as e:
        print("Paired sample t-test failed.", str(e))
        return None, None

    print(f"Paired sample t-test t-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    return t_stat, p_value


# Wilcoxon signed-rank test
# WHEN: Both samples are not normally distributed
def perform_wilcoxon_signed_rank_test(data1, data2):
    """
    Perform a Wilcoxon signed-rank test on two sets of data.

    Parameters:
    data1 (array-like): First set of data
    data2 (array-like): Second set of data

    Returns:
    statistic (float): Test statistic
    p_value (float): P-value
    """
    try:
        statistic, p_value = stats.wilcoxon(data1, data2)
    except ValueError as e:
        print("Wilcoxon signed-rank test failed.", str(e))
        return None, None

    print(f"Wilcoxon signed-rank test statistic: {statistic}")
    print(f"P-value: {p_value}")

    return statistic, p_value


def interpretation(p_value, alpha=0.05):
    """
    Interpret the p-value of a statistical test.

    Parameters:
    p_value (float): P-value of the test
    alpha (float): Significance level

    Returns:
    interpretation (str): Interpretation of the test result
    """
    if p_value < alpha:
        print(
            "The difference in scores between the two runs is statistically significant."
        )
    else:
        print(
            "The difference in scores between the two runs is not statistically significant."
        )


# -----------------------------------


def complete_statistical_testing(
    output1, output2, runid, perform_paired_t_test_only=True
):
    print(f"\n\nComparing {runid} with baseline-{runid}:")
    scores1 = parse_scores(output1)
    scores2 = parse_scores(output2)

    p_value = None

    if perform_paired_t_test_only:
        t_stat, p_value = perform_paired_sample_t_test(scores1, scores2)
    else:
        # Perform Shapiro-Wilk test for normality
        stat1, p_value1 = perform_shapiro_wilk_test(scores1)
        stat2, p_value2 = perform_shapiro_wilk_test(scores2)

        alpha = 0.05
        if p_value1 > alpha and p_value2 > alpha:
            print("Both samples are normally distributed.")
        else:
            print("At least one of the samples is not normally distributed.")

        # Plot for Run 1
        plot_qq_hist(scores1, "baseline-" + runid)

        # Plot for Run 2
        plot_qq_hist(scores2, runid)

        print()

        if p_value1 > alpha and p_value2 > alpha:
            # Both samples are normally distributed
            t_stat, p_value = perform_paired_sample_t_test(scores1, scores2)
        else:
            # At least one sample is not normally distributed
            stat, p_value = perform_wilcoxon_signed_rank_test(scores1, scores2)

    if p_value is not None:
        interpretation(p_value, alpha=0.05)
