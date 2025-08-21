import math
import argparse

def calculate_approximate_ci(score_pct, n):
    """
    Calculates an approximate 95% confidence interval margin of error for a
    percentage-based score (e.g., F1-score, Accuracy).

    This uses the standard error of a proportion as a robust approximation.

    Args:
        score_pct (float): The observed score as a percentage (e.g., 60.62).
        n (int): The total number of test samples.

    Returns:
        float: The margin of error as a percentage.
    """
    if n == 0 or score_pct < 0 or score_pct > 100:
        return 0

    x = (score_pct/100)*n
    x_tilda = x+2
    n_tilda = n + 4
    p = score_pct/100.0
    p_tilda = x_tilda/n_tilda
    z = 1.96  # z-score for 95% confidence

    # Standard error of a proportion: sqrt(p * (1-p) / n)
    se = math.sqrt(p_tilda * (1 - p_tilda) / n_tilda)
    margin_of_error = z * se

    # Return as a percentage
    return margin_of_error * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate confidence interval")
    parser.add_argument("--n", type=int, required=True, help="number of test samples")
    parser.add_argument("--score_pct", type=float, required=True, help="metric score")
    args = parser.parse_args()
    print(calculate_approximate_ci(args.score_pct, args.n))