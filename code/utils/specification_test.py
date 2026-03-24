"""
Specification test for choosing between L1 and L2 HP filters.

Implements a Hausman-type test and an asymmetry pre-test to guide
practitioners on whether the L1 (robust) HP filter offers a statistically
significant improvement over the standard L2 HP filter for a given series.

The core idea: under symmetric DGPs the two filters should produce similar
trends. Large divergence between tau_L1 and tau_L2 is evidence of
asymmetry that the L1 filter can exploit.
"""

import numpy as np
from scipy import stats
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from filters.hp import l1_hp_filter, l2_hp_filter
from utils.tests import AsymmetryTests


def hausman_test_l1_l2(y, lamb_l1=600, lamb_l2=1600, n_bootstrap=500):
    """
    Hausman-type test: H0: L1 and L2 give same trend (symmetry).
    Bootstrap the distribution of ||tau_L1 - tau_L2|| under H0.

    Under H0 (symmetric DGP), generate bootstrap samples from L2 residuals,
    apply both filters, compute ||tau_L1 - tau_L2||.
    Compare observed statistic to bootstrap distribution.

    Parameters
    ----------
    y : array-like
        Input time series.
    lamb_l1 : float
        Smoothing parameter for the L1 HP filter (default 600).
    lamb_l2 : float
        Smoothing parameter for the L2 HP filter (default 1600).
    n_bootstrap : int
        Number of bootstrap replications (default 500).

    Returns
    -------
    dict
        'test_statistic' : float
            Observed RMSD between L1 and L2 trends.
        'p_value' : float
            Bootstrap p-value (fraction of bootstrap stats >= observed).
        'reject_h0' : bool
            Whether H0 is rejected at the 5% level.
        'critical_value_5pct' : float
            95th percentile of the bootstrap distribution.
        'bootstrap_distribution' : np.ndarray
            Full array of bootstrap test statistics.
    """
    y = np.asarray(y, dtype=float).flatten()
    T = len(y)

    # Step 1: Compute observed trends on original data
    tau_l1, _ = l1_hp_filter(y, lamb=lamb_l1)
    tau_l2, cycle_l2 = l2_hp_filter(y, lamb=lamb_l2)

    # Step 2: Observed test statistic = RMSD(tau_L1, tau_L2)
    test_stat = np.sqrt(np.mean((tau_l1 - tau_l2) ** 2))

    # Step 3: Bootstrap under H0 (symmetric DGP)
    # Under H0, the L2 decomposition is correct; resample the L2 cycle
    # using a block bootstrap to preserve serial correlation.
    block_size = 8
    bootstrap_stats = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Block bootstrap of the L2 cycle
        cycle_boot = _block_bootstrap(cycle_l2, block_size=block_size)

        # Construct bootstrap series: L2 trend + resampled cycle
        y_boot = tau_l2 + cycle_boot

        # Apply both filters to the bootstrap series
        try:
            tau_l1_boot, _ = l1_hp_filter(y_boot, lamb=lamb_l1)
        except RuntimeError:
            # If L1 solver fails on a bootstrap draw, use NaN and skip
            bootstrap_stats[b] = np.nan
            continue
        tau_l2_boot, _ = l2_hp_filter(y_boot, lamb=lamb_l2)

        # Bootstrap test statistic
        bootstrap_stats[b] = np.sqrt(np.mean((tau_l1_boot - tau_l2_boot) ** 2))

    # Drop any NaN entries from failed solves
    valid_stats = bootstrap_stats[~np.isnan(bootstrap_stats)]

    if len(valid_stats) == 0:
        return {
            'test_statistic': test_stat,
            'p_value': np.nan,
            'reject_h0': False,
            'critical_value_5pct': np.nan,
            'bootstrap_distribution': bootstrap_stats,
        }

    # Step 4: p-value = fraction of bootstrap stats >= observed
    p_value = np.mean(valid_stats >= test_stat)
    critical_value_5pct = np.percentile(valid_stats, 95)

    return {
        'test_statistic': test_stat,
        'p_value': p_value,
        'reject_h0': p_value < 0.05,
        'critical_value_5pct': critical_value_5pct,
        'bootstrap_distribution': valid_stats,
    }


def pretest_asymmetry(y, lamb_l2=1600, alpha=0.05):
    """
    Pre-test for asymmetry on L2-HP cycle residuals.
    Run skewness, triples, and Sichel tests.
    Recommend L1 if ANY test rejects at alpha.

    Parameters
    ----------
    y : array-like
        Input time series.
    lamb_l2 : float
        Smoothing parameter for the L2 HP filter (default 1600).
    alpha : float
        Significance level for rejection (default 0.05).

    Returns
    -------
    dict
        'skewness_test' : dict
            Results from AsymmetryTests.skewness_test().
        'triples_test' : dict
            Results from AsymmetryTests.triples_test().
        'sichel_test' : dict
            Results from AsymmetryTests.sichel_test().
        'any_reject' : bool
            True if any test rejects H0 of symmetry at the given alpha.
        'recommendation' : str
            'L1' if any_reject is True, 'L2' otherwise.
        'alpha' : float
            Significance level used.
    """
    y = np.asarray(y, dtype=float).flatten()

    # Extract L2-HP cycle
    _, cycle_l2 = l2_hp_filter(y, lamb=lamb_l2)

    # Run asymmetry tests on the cycle
    skew_result = AsymmetryTests.skewness_test(cycle_l2, bootstrap_samples=1000)
    triples_result = AsymmetryTests.triples_test(cycle_l2)
    sichel_result = AsymmetryTests.sichel_test(cycle_l2)

    # Check rejections at the specified alpha level
    skew_reject = skew_result.get('p_value_analytical', 1.0) < alpha
    triples_reject = triples_result.get('p_value', 1.0) < alpha
    deepness_reject = sichel_result.get('deepness_p', 1.0) < alpha
    steepness_reject = sichel_result.get('steepness_p', 1.0) < alpha

    any_reject = skew_reject or triples_reject or deepness_reject or steepness_reject

    return {
        'skewness_test': skew_result,
        'triples_test': triples_result,
        'sichel_test': sichel_result,
        'skew_reject': skew_reject,
        'triples_reject': triples_reject,
        'deepness_reject': deepness_reject,
        'steepness_reject': steepness_reject,
        'any_reject': any_reject,
        'recommendation': 'L1' if any_reject else 'L2',
        'alpha': alpha,
    }


def recommend_filter(y, lamb_l1=600, lamb_l2=1600, alpha=0.05, n_bootstrap=200):
    """
    Full decision function combining pretest + Hausman test.

    The procedure works in two stages:
    1. Pre-test: check whether the L2-HP cycle exhibits statistically
       significant asymmetry (skewness, triples, or Sichel tests).
    2. Hausman test: bootstrap whether the divergence between L1 and L2
       trends is larger than expected under symmetric null.

    L1 is recommended if EITHER stage finds evidence of asymmetry.

    Parameters
    ----------
    y : array-like
        Input time series.
    lamb_l1 : float
        Smoothing parameter for L1 HP filter (default 600).
    lamb_l2 : float
        Smoothing parameter for L2 HP filter (default 1600).
    alpha : float
        Significance level (default 0.05).
    n_bootstrap : int
        Number of bootstrap replications for Hausman test (default 200).

    Returns
    -------
    dict
        'recommendation' : str
            'L1' or 'L2'.
        'pretest' : dict
            Pre-test results from pretest_asymmetry().
        'hausman' : dict
            Hausman test results from hausman_test_l1_l2().
        'summary' : str
            Human-readable summary string.
    """
    y = np.asarray(y, dtype=float).flatten()

    # Stage 1: Pre-test for asymmetry
    pretest = pretest_asymmetry(y, lamb_l2=lamb_l2, alpha=alpha)

    # Stage 2: Hausman-type test
    hausman = hausman_test_l1_l2(
        y, lamb_l1=lamb_l1, lamb_l2=lamb_l2, n_bootstrap=n_bootstrap
    )

    # Decision: recommend L1 if either test flags asymmetry
    pretest_recommends_l1 = pretest['any_reject']
    hausman_recommends_l1 = hausman['reject_h0']
    recommend_l1 = pretest_recommends_l1 or hausman_recommends_l1

    recommendation = 'L1' if recommend_l1 else 'L2'

    # Build human-readable summary
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("  Specification Test: L1 vs L2 HP Filter")
    summary_lines.append("=" * 60)
    summary_lines.append("")

    # Pre-test summary
    summary_lines.append("Stage 1: Asymmetry Pre-test on L2-HP Cycle")
    summary_lines.append("-" * 45)
    skew_p = pretest['skewness_test'].get('p_value_analytical', np.nan)
    trip_p = pretest['triples_test'].get('p_value', np.nan)
    deep_p = pretest['sichel_test'].get('deepness_p', np.nan)
    steep_p = pretest['sichel_test'].get('steepness_p', np.nan)

    summary_lines.append(
        f"  Skewness test:  p = {skew_p:.4f}  {'[REJECT]' if pretest['skew_reject'] else ''}"
    )
    summary_lines.append(
        f"  Triples test:   p = {trip_p:.4f}  {'[REJECT]' if pretest['triples_reject'] else ''}"
    )
    summary_lines.append(
        f"  Sichel deepness: p = {deep_p:.4f}  {'[REJECT]' if pretest['deepness_reject'] else ''}"
    )
    summary_lines.append(
        f"  Sichel steepness: p = {steep_p:.4f}  {'[REJECT]' if pretest['steepness_reject'] else ''}"
    )
    summary_lines.append(
        f"  Pre-test recommendation: {pretest['recommendation']}"
    )
    summary_lines.append("")

    # Hausman test summary
    summary_lines.append("Stage 2: Hausman-type Test (L1 vs L2 trend divergence)")
    summary_lines.append("-" * 55)
    summary_lines.append(
        f"  Test statistic (RMSD): {hausman['test_statistic']:.6f}"
    )
    summary_lines.append(
        f"  Bootstrap p-value:     {hausman['p_value']:.4f}"
    )
    summary_lines.append(
        f"  Critical value (5%):   {hausman['critical_value_5pct']:.6f}"
    )
    summary_lines.append(
        f"  Reject H0 (symmetry):  {hausman['reject_h0']}"
    )
    summary_lines.append("")

    # Final recommendation
    summary_lines.append("=" * 60)
    summary_lines.append(f"  RECOMMENDATION: Use {recommendation} HP filter")
    if recommend_l1:
        reasons = []
        if pretest_recommends_l1:
            reasons.append("asymmetry detected in cycle residuals")
        if hausman_recommends_l1:
            reasons.append("significant L1-L2 trend divergence")
        summary_lines.append(f"  Reason: {'; '.join(reasons)}")
    else:
        summary_lines.append(
            "  Reason: no evidence of asymmetry; L2 filter is sufficient"
        )
    summary_lines.append("=" * 60)

    summary = "\n".join(summary_lines)

    return {
        'recommendation': recommendation,
        'pretest': pretest,
        'hausman': hausman,
        'summary': summary,
    }


def _block_bootstrap(x, block_size=8):
    """
    Non-overlapping block bootstrap for a 1-D array.

    Resamples blocks of consecutive observations (with replacement)
    to preserve serial correlation structure in the resampled series.

    Parameters
    ----------
    x : np.ndarray
        Input array of length T.
    block_size : int
        Length of each block (default 8, ~2 years of quarterly data).

    Returns
    -------
    np.ndarray
        Resampled array of the same length as x.
    """
    T = len(x)
    n_blocks = int(np.ceil(T / block_size))

    # Number of possible starting positions for blocks
    max_start = T - block_size
    if max_start < 1:
        # Series too short for block bootstrap; fall back to iid resample
        return np.random.choice(x, size=T, replace=True)

    # Draw random block starting indices
    starts = np.random.randint(0, max_start + 1, size=n_blocks)

    # Concatenate blocks and truncate to length T
    blocks = [x[s : s + block_size] for s in starts]
    resampled = np.concatenate(blocks)[:T]

    return resampled
