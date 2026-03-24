"""
Bootstrap Inference for L1-HP Filter

This module implements block bootstrap methods for uncertainty quantification
of L1-HP trend and cycle estimates. Block bootstrap preserves the serial
correlation structure of macroeconomic time series.

References:
- Kunsch (1989): The Jackknife and Bootstrap for General Stationary Observations
- Politis & Romano (1994): The Stationary Bootstrap
- Politis & White (2004): Automatic Block-Length Selection
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import sys
import os

# Ensure correct path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter


# =============================================================================
# Block Length Selection
# =============================================================================

def optimal_block_length_ar1(y):
    """
    Estimate optimal block length based on AR(1) approximation.

    Uses the rule of thumb: b = T^(1/3) * (2 * rho^2 / (1 - rho^2))^(1/3)
    where rho is the AR(1) coefficient.

    Parameters
    ----------
    y : array-like
        Time series data

    Returns
    -------
    int
        Recommended block length
    """
    y = np.asarray(y).flatten()
    T = len(y)

    # Estimate AR(1) coefficient
    y_lag = y[:-1]
    y_current = y[1:]
    rho = np.corrcoef(y_lag, y_current)[0, 1]

    # Rule of thumb for block length (Lahiri, 2003)
    if abs(rho) < 0.99:
        factor = (2 * rho**2 / (1 - rho**2))**(1/3)
    else:
        factor = 2  # High persistence case

    b = int(np.ceil(T**(1/3) * max(factor, 1)))

    # Ensure reasonable bounds
    b = max(4, min(b, T // 4))

    return b


def optimal_block_length_autocorr(y, max_lag=20):
    """
    Estimate optimal block length based on autocorrelation decay.

    Block length chosen as the lag at which autocorrelation drops below
    2/sqrt(T) (approximate 95% significance threshold).

    Parameters
    ----------
    y : array-like
        Time series data
    max_lag : int
        Maximum lag to consider

    Returns
    -------
    int
        Recommended block length
    """
    y = np.asarray(y).flatten()
    T = len(y)
    threshold = 2 / np.sqrt(T)

    # Compute autocorrelations
    y_centered = y - np.mean(y)
    var_y = np.var(y)

    for lag in range(1, min(max_lag + 1, T // 4)):
        autocorr = np.mean(y_centered[lag:] * y_centered[:-lag]) / var_y
        if abs(autocorr) < threshold:
            return max(4, lag)

    # If autocorrelation doesn't decay, use rule of thumb
    return max(4, int(np.ceil(T**(1/3))))


# =============================================================================
# Block Bootstrap Methods
# =============================================================================

def moving_block_bootstrap(residuals, block_length, n_boot=1):
    """
    Generate bootstrap samples using Moving Block Bootstrap (MBB).

    Parameters
    ----------
    residuals : array-like
        Original residuals (cycle estimates)
    block_length : int
        Length of each block
    n_boot : int
        Number of bootstrap samples to generate

    Returns
    -------
    array
        Bootstrap residual samples (n_boot x T)
    """
    residuals = np.asarray(residuals).flatten()
    T = len(residuals)
    b = block_length

    # Number of blocks needed to cover T observations
    n_blocks = int(np.ceil(T / b))

    boot_samples = np.zeros((n_boot, T))

    for i in range(n_boot):
        # Randomly select block starting points
        start_indices = np.random.randint(0, T - b + 1, size=n_blocks)

        # Concatenate blocks
        sample = []
        for start in start_indices:
            sample.extend(residuals[start:start + b])

        # Trim to length T
        boot_samples[i, :] = sample[:T]

    return boot_samples


def circular_block_bootstrap(residuals, block_length, n_boot=1):
    """
    Generate bootstrap samples using Circular Block Bootstrap.

    The series is wrapped around (circular), avoiding edge effects.

    Parameters
    ----------
    residuals : array-like
        Original residuals (cycle estimates)
    block_length : int
        Length of each block
    n_boot : int
        Number of bootstrap samples to generate

    Returns
    -------
    array
        Bootstrap residual samples (n_boot x T)
    """
    residuals = np.asarray(residuals).flatten()
    T = len(residuals)
    b = block_length

    # Extend residuals circularly
    residuals_circular = np.concatenate([residuals, residuals[:b-1]])

    n_blocks = int(np.ceil(T / b))
    boot_samples = np.zeros((n_boot, T))

    for i in range(n_boot):
        # Randomly select block starting points (any position in original series)
        start_indices = np.random.randint(0, T, size=n_blocks)

        # Concatenate blocks
        sample = []
        for start in start_indices:
            sample.extend(residuals_circular[start:start + b])

        boot_samples[i, :] = sample[:T]

    return boot_samples


# =============================================================================
# Main Bootstrap Functions
# =============================================================================

def block_bootstrap_l1_hp(y, lamb=600, n_boot=500, alpha=0.90,
                          block_length=None, method='circular',
                          show_progress=True):
    """
    Generate confidence intervals for L1-HP trend using Block Bootstrap.

    This method preserves the serial correlation structure of the cycle
    by resampling blocks of residuals rather than individual observations.

    Parameters
    ----------
    y : array-like
        Time series data
    lamb : float
        L1-HP smoothing parameter
    n_boot : int
        Number of bootstrap replications
    alpha : float
        Confidence level (e.g., 0.90 for 90% CI)
    block_length : int, optional
        Block length for bootstrap. If None, automatically selected.
    method : str
        'circular' (default) or 'moving' block bootstrap
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    dict with keys:
        'trend': Original trend estimate
        'cycle': Original cycle estimate
        'lower': Lower confidence bound for trend
        'upper': Upper confidence bound for trend
        'cycle_lower': Lower confidence bound for cycle
        'cycle_upper': Upper confidence bound for cycle
        'trends_boot': All bootstrap trend estimates (n_boot x T)
        'block_length': Block length used
    """
    y = np.asarray(y).flatten()
    T = len(y)

    # 1. Initial fit
    trend_hat, cycle_hat = l1_hp_filter(y, lamb=lamb)

    # 2. Determine block length
    if block_length is None:
        block_length = optimal_block_length_ar1(cycle_hat)

    # 3. Generate bootstrap samples
    if method == 'circular':
        boot_residuals = circular_block_bootstrap(cycle_hat, block_length, n_boot)
    else:
        boot_residuals = moving_block_bootstrap(cycle_hat, block_length, n_boot)

    # 4. Re-estimate trend for each bootstrap sample
    trends_boot = np.zeros((n_boot, T))
    cycles_boot = np.zeros((n_boot, T))

    iterator = range(n_boot)
    if show_progress:
        iterator = tqdm(iterator, desc="Bootstrap")

    for i in iterator:
        # Synthetic data under null: trend is fixed, resample cycle
        y_star = trend_hat + boot_residuals[i, :]

        try:
            trend_star, cycle_star = l1_hp_filter(y_star, lamb=lamb)
            trends_boot[i, :] = trend_star
            cycles_boot[i, :] = cycle_star
        except Exception:
            # If optimization fails, use original
            trends_boot[i, :] = trend_hat
            cycles_boot[i, :] = cycle_hat

    # 5. Compute confidence intervals
    lower_p = (1 - alpha) / 2 * 100
    upper_p = (1 + alpha) / 2 * 100

    trend_lower = np.percentile(trends_boot, lower_p, axis=0)
    trend_upper = np.percentile(trends_boot, upper_p, axis=0)

    cycle_lower = np.percentile(cycles_boot, lower_p, axis=0)
    cycle_upper = np.percentile(cycles_boot, upper_p, axis=0)

    return {
        'trend': trend_hat,
        'cycle': cycle_hat,
        'lower': trend_lower,
        'upper': trend_upper,
        'cycle_lower': cycle_lower,
        'cycle_upper': cycle_upper,
        'trends_boot': trends_boot,
        'cycles_boot': cycles_boot,
        'block_length': block_length,
        'n_boot': n_boot,
        'alpha': alpha
    }


def bootstrap_cycle_at_date(y, dates, target_dates, lamb=600, n_boot=500,
                            alpha=0.90, block_length=None):
    """
    Compute bootstrap confidence intervals for cycle at specific dates.

    Useful for reporting uncertainty at recession troughs.

    Parameters
    ----------
    y : array-like
        Time series data
    dates : array-like
        Date index for y
    target_dates : list of str or Timestamp
        Dates at which to report CIs
    lamb : float
        L1-HP smoothing parameter
    n_boot : int
        Number of bootstrap replications
    alpha : float
        Confidence level
    block_length : int, optional
        Block length for bootstrap

    Returns
    -------
    pd.DataFrame
        Cycle estimates and CIs at target dates
    """
    # Run full bootstrap
    results = block_bootstrap_l1_hp(y, lamb=lamb, n_boot=n_boot,
                                     alpha=alpha, block_length=block_length)

    dates = pd.DatetimeIndex(dates)

    output = []
    for target in target_dates:
        target_dt = pd.Timestamp(target)

        # Find closest date
        idx = dates.get_indexer([target_dt], method='nearest')[0]

        if idx >= 0 and idx < len(y):
            cycle_point = results['cycle'][idx] * 100
            cycle_lo = results['cycle_lower'][idx] * 100
            cycle_hi = results['cycle_upper'][idx] * 100
            cycle_se = np.std(results['cycles_boot'][:, idx]) * 100

            output.append({
                'date': dates[idx],
                'cycle_pct': cycle_point,
                'ci_lower': cycle_lo,
                'ci_upper': cycle_hi,
                'std_error': cycle_se
            })

    return pd.DataFrame(output)


# =============================================================================
# Coverage Validation
# =============================================================================

def validate_coverage(dgp, lamb=600, n_sim=200, n_boot=200, alpha=0.90,
                      T=200, show_progress=True):
    """
    Validate bootstrap coverage via Monte Carlo simulation.

    For a known DGP, checks whether the (1-alpha)% confidence intervals
    actually contain the true trend (1-alpha)% of the time.

    Parameters
    ----------
    dgp : DGP object
        Data generating process with generate(T, seed) method
    lamb : float
        L1-HP smoothing parameter
    n_sim : int
        Number of Monte Carlo simulations
    n_boot : int
        Number of bootstrap replications per simulation
    alpha : float
        Nominal confidence level
    T : int
        Time series length
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    dict with coverage statistics
    """
    coverage_trend = []
    coverage_cycle = []
    interval_widths = []

    iterator = range(n_sim)
    if show_progress:
        iterator = tqdm(iterator, desc="Coverage validation")

    for sim in iterator:
        # Generate data from known DGP
        y, true_trend, true_cycle = dgp.generate(T, seed=sim)

        # Run bootstrap
        try:
            results = block_bootstrap_l1_hp(
                y, lamb=lamb, n_boot=n_boot, alpha=alpha, show_progress=False
            )

            # Check coverage: what fraction of time points are covered?
            trend_covered = np.mean(
                (results['lower'] <= true_trend) & (true_trend <= results['upper'])
            )
            cycle_covered = np.mean(
                (results['cycle_lower'] <= true_cycle) & (true_cycle <= results['cycle_upper'])
            )

            coverage_trend.append(trend_covered)
            coverage_cycle.append(cycle_covered)

            # Average interval width
            width = np.mean(results['upper'] - results['lower'])
            interval_widths.append(width)

        except Exception as e:
            print(f"  Simulation {sim} failed: {e}")
            continue

    return {
        'nominal_coverage': alpha,
        'trend_coverage_mean': np.mean(coverage_trend),
        'trend_coverage_std': np.std(coverage_trend),
        'cycle_coverage_mean': np.mean(coverage_cycle),
        'cycle_coverage_std': np.std(coverage_cycle),
        'avg_interval_width': np.mean(interval_widths),
        'n_simulations': len(coverage_trend)
    }


def run_coverage_validation_suite(output_dir=None, n_sim=200, n_boot=200, T=200):
    """
    Run coverage validation across multiple DGPs.

    Parameters
    ----------
    output_dir : str
        Directory to save results
    n_sim : int
        Number of simulations per DGP
    n_boot : int
        Bootstrap replications per simulation
    T : int
        Time series length

    Returns
    -------
    pd.DataFrame
        Coverage results by DGP
    """
    from models.dgps import (
        UCModelDGP, SkewedUCDGP, StatisticalPluckingDGP, SharpCrashDGP
    )

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # DGPs to test
    dgps = {
        'UCModel': UCModelDGP(),
        'SkewedUC': SkewedUCDGP(),
        'Plucking': StatisticalPluckingDGP(),
        'SharpCrash': SharpCrashDGP()
    }

    print("\n" + "="*70)
    print("BOOTSTRAP COVERAGE VALIDATION")
    print("="*70)
    print(f"Settings: n_sim={n_sim}, n_boot={n_boot}, T={T}, alpha=0.90")

    results = []

    for name, dgp in dgps.items():
        print(f"\n--- {name} ---")

        coverage = validate_coverage(
            dgp, lamb=600, n_sim=n_sim, n_boot=n_boot,
            alpha=0.90, T=T, show_progress=True
        )

        coverage['DGP'] = name
        results.append(coverage)

        print(f"  Trend coverage: {coverage['trend_coverage_mean']:.1%} "
              f"(nominal: {coverage['nominal_coverage']:.0%})")
        print(f"  Cycle coverage: {coverage['cycle_coverage_mean']:.1%}")

    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[['DGP', 'nominal_coverage', 'trend_coverage_mean',
                             'trend_coverage_std', 'cycle_coverage_mean',
                             'cycle_coverage_std', 'avg_interval_width', 'n_simulations']]

    # Save results
    results_df.to_csv(os.path.join(output_dir, 'bootstrap_coverage.csv'), index=False)

    print("\n" + "="*70)
    print("COVERAGE SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))

    return results_df


# =============================================================================
# Legacy function (for backwards compatibility)
# =============================================================================

def bootstrap_l1_hp(y, lamb=600, n_boot=100, alpha=0.90):
    """
    Legacy function - now calls block_bootstrap_l1_hp.

    Kept for backwards compatibility.
    """
    results = block_bootstrap_l1_hp(y, lamb=lamb, n_boot=n_boot,
                                     alpha=alpha, show_progress=False)
    return results['trend'], results['lower'], results['upper'], results['trends_boot']


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bootstrap Coverage Validation')
    parser.add_argument('--n_sim', type=int, default=200,
                       help='Number of simulations')
    parser.add_argument('--n_boot', type=int, default=200,
                       help='Bootstrap replications')
    parser.add_argument('--T', type=int, default=200,
                       help='Time series length')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with reduced settings')
    args = parser.parse_args()

    if args.quick:
        n_sim, n_boot, T = 50, 100, 100
    else:
        n_sim, n_boot, T = args.n_sim, args.n_boot, args.T

    results = run_coverage_validation_suite(n_sim=n_sim, n_boot=n_boot, T=T)
