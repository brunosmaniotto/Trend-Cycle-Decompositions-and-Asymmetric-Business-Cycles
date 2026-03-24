"""
Hamilton (2018) Replication Experiments

Replicates key experiments from:
- Hamilton (2018) "Why You Should Never Use the Hodrick-Prescott Filter"
- Hodrick (2020) "An Exploration of Trend-Cycle Decomposition Methodologies"

Extends analysis to compare L1 vs L2 versions of each filter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

from filters.hp import l2_hp_filter, l1_hp_filter
from filters.hamilton import l2_hamilton_filter, l1_hamilton_filter


def generate_random_walk(T, mu=0.0, sigma=1.0, seed=None):
    """Generate a pure random walk."""
    if seed is not None:
        np.random.seed(seed)
    innovations = np.random.normal(mu, sigma, T)
    return np.cumsum(innovations)


def generate_arima(T, ar_coefs=[0.9], ma_coefs=[], d=1, mu=0.0, sigma=1.0, seed=None):
    """Generate ARIMA process."""
    if seed is not None:
        np.random.seed(seed)

    # Generate stationary AR component
    p = len(ar_coefs)
    q = len(ma_coefs)

    eps = np.random.normal(0, sigma, T + 100)
    y_stationary = np.zeros(T + 100)

    for t in range(max(p, q), T + 100):
        ar_part = sum(ar_coefs[i] * y_stationary[t-i-1] for i in range(p))
        ma_part = sum(ma_coefs[i] * eps[t-i-1] for i in range(q))
        y_stationary[t] = ar_part + eps[t] + ma_part

    y_stationary = y_stationary[100:]  # Remove burn-in

    # Add drift
    y_stationary += mu * np.arange(T)

    # Integrate d times
    y = y_stationary.copy()
    for _ in range(d):
        y = np.cumsum(y)

    return y


def generate_uc_model(T, trend_sigma=0.01, cycle_sigma=0.02, cycle_ar=0.9, seed=None):
    """
    Generate Unobserved Components model.
    y_t = tau_t + c_t
    tau_t = tau_{t-1} + eta_t  (random walk trend)
    c_t = rho * c_{t-1} + eps_t (AR(1) cycle)
    """
    if seed is not None:
        np.random.seed(seed)

    # Trend component (random walk)
    trend = np.cumsum(np.random.normal(0, trend_sigma, T))

    # Cycle component (AR(1))
    cycle = np.zeros(T)
    for t in range(1, T):
        cycle[t] = cycle_ar * cycle[t-1] + np.random.normal(0, cycle_sigma)

    return trend + cycle, trend, cycle


def spurious_cycle_test(filter_func, filter_name, n_sims=500, T=200, seed=42):
    """
    Hamilton's spurious cycle test.

    Apply filter to random walk data (which has no true cycle).
    Measure properties of the "spurious" cycle that emerges.

    Parameters
    ----------
    filter_func : callable
        Filter function returning (trend, cycle)
    filter_name : str
        Name for reporting
    n_sims : int
        Number of Monte Carlo replications
    T : int
        Sample size
    seed : int
        Random seed

    Returns
    -------
    dict with spurious cycle statistics
    """
    np.random.seed(seed)

    cycle_vars = []
    cycle_autocorrs = []
    cycle_skews = []

    for i in range(n_sims):
        # Generate random walk
        y = generate_random_walk(T, mu=0.005, sigma=0.01)

        try:
            _, cycle = filter_func(y)

            # Filter out NaN values (Hamilton filters have NaN at start)
            cycle_valid = cycle[~np.isnan(cycle)]

            # Skip if no valid values or zero variance
            if len(cycle_valid) == 0 or np.std(cycle_valid) == 0:
                continue

            cycle_vars.append(np.var(cycle_valid))

            # First-order autocorrelation (use valid values)
            if len(cycle_valid) > 1:
                autocorr = np.corrcoef(cycle_valid[:-1], cycle_valid[1:])[0, 1]
                cycle_autocorrs.append(autocorr)

            cycle_skews.append(stats.skew(cycle_valid))

        except Exception:
            continue

    results = {
        'filter': filter_name,
        'mean_cycle_var': np.mean(cycle_vars) if cycle_vars else np.nan,
        'mean_cycle_std': np.sqrt(np.mean(cycle_vars)) if cycle_vars else np.nan,
        'mean_autocorr': np.mean(cycle_autocorrs) if cycle_autocorrs else np.nan,
        'mean_skewness': np.mean(cycle_skews) if cycle_skews else np.nan,
        'n_successful': len(cycle_vars)
    }

    return results


def trend_accuracy_test(filter_func, filter_name, dgp_func, dgp_name,
                        n_sims=500, T=200, seed=42):
    """
    Test filter accuracy in recovering true trend.

    Parameters
    ----------
    filter_func : callable
        Filter function
    filter_name : str
        Filter name
    dgp_func : callable
        Function that returns (y, true_trend, true_cycle)
    dgp_name : str
        DGP name
    n_sims : int
        Number of simulations
    T : int
        Sample size
    seed : int
        Random seed

    Returns
    -------
    dict with accuracy statistics
    """
    np.random.seed(seed)

    trend_rmses = []
    cycle_rmses = []
    cycle_corrs = []

    for i in range(n_sims):
        y, true_trend, true_cycle = dgp_func(T, seed=seed + i)

        try:
            est_trend, est_cycle = filter_func(y)

            # Get valid (non-NaN) indices for Hamilton filters
            valid_mask = ~np.isnan(est_trend)

            # Skip if no valid values
            if not np.any(valid_mask):
                continue

            # Extract valid portions
            est_trend_valid = est_trend[valid_mask]
            est_cycle_valid = est_cycle[valid_mask]
            true_trend_valid = true_trend[valid_mask]
            true_cycle_valid = true_cycle[valid_mask]

            # RMSE for trend (on valid portion)
            trend_rmse = np.sqrt(np.mean((est_trend_valid - true_trend_valid)**2))
            trend_rmses.append(trend_rmse)

            # RMSE for cycle (on valid portion)
            cycle_rmse = np.sqrt(np.mean((est_cycle_valid - true_cycle_valid)**2))
            cycle_rmses.append(cycle_rmse)

            # Correlation with true cycle (on valid portion)
            if np.std(est_cycle_valid) > 0 and np.std(true_cycle_valid) > 0:
                corr = np.corrcoef(est_cycle_valid, true_cycle_valid)[0, 1]
                cycle_corrs.append(corr)

        except Exception:
            continue

    results = {
        'filter': filter_name,
        'dgp': dgp_name,
        'trend_rmse': np.mean(trend_rmses) if trend_rmses else np.nan,
        'cycle_rmse': np.mean(cycle_rmses) if cycle_rmses else np.nan,
        'cycle_corr': np.mean(cycle_corrs) if cycle_corrs else np.nan,
        'n_successful': len(trend_rmses)
    }

    return results


def run_hamilton_experiments(output_dir=None, n_sims=500, lam=1600):
    """
    Run full suite of Hamilton (2018) replication experiments.

    Parameters
    ----------
    output_dir : str, optional
        Output directory
    n_sims : int
        Number of Monte Carlo simulations
    lam : float
        HP filter smoothing parameter
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'experiments')
    os.makedirs(output_dir, exist_ok=True)

    # Define filters
    filters = [
        (lambda x: l2_hp_filter(x, lamb=lam), 'HP-L2'),
        (lambda x: l1_hp_filter(x, lamb=lam), 'HP-L1'),
        (lambda x: l2_hamilton_filter(x), 'Hamilton-L2'),
        (lambda x: l1_hamilton_filter(x), 'Hamilton-L1')
    ]

    print("=" * 70)
    print("HAMILTON (2018) REPLICATION EXPERIMENTS")
    print("=" * 70)

    # 1. Spurious Cycle Test (Hamilton's main critique)
    print("\n1. SPURIOUS CYCLE TEST")
    print("   Applying filters to random walk data (no true cycle exists)")
    print("-" * 70)

    spurious_results = []
    for func, name in filters:
        result = spurious_cycle_test(func, name, n_sims=n_sims)
        spurious_results.append(result)
        print(f"\n{name}:")
        print(f"  Spurious cycle std:     {result['mean_cycle_std']:.4f}")
        print(f"  Spurious autocorr:      {result['mean_autocorr']:.3f}")
        print(f"  Spurious skewness:      {result['mean_skewness']:.3f}")

    spurious_df = pd.DataFrame(spurious_results)
    spurious_df.to_csv(os.path.join(output_dir, 'hamilton_spurious_cycle.csv'), index=False)

    # 2. Trend Accuracy Test (Hodrick's defense - UC model)
    print("\n\n2. TREND ACCURACY TEST (UC Model)")
    print("   Testing accuracy when true trend/cycle exists")
    print("-" * 70)

    def uc_dgp(T, seed=None):
        return generate_uc_model(T, trend_sigma=0.01, cycle_sigma=0.02, cycle_ar=0.9, seed=seed)

    accuracy_results = []
    for func, name in filters:
        result = trend_accuracy_test(func, name, uc_dgp, 'UC-Model', n_sims=n_sims)
        accuracy_results.append(result)
        print(f"\n{name}:")
        print(f"  Trend RMSE:    {result['trend_rmse']:.4f}")
        print(f"  Cycle RMSE:    {result['cycle_rmse']:.4f}")
        print(f"  Cycle Corr:    {result['cycle_corr']:.3f}")

    accuracy_df = pd.DataFrame(accuracy_results)
    accuracy_df.to_csv(os.path.join(output_dir, 'hamilton_accuracy_test.csv'), index=False)

    # 3. Asymmetric DGP Test (our extension)
    print("\n\n3. ASYMMETRIC DGP TEST (Our Extension)")
    print("   Testing performance on asymmetric plucking model")
    print("-" * 70)

    def plucking_dgp(T, seed=None):
        """Plucking model with asymmetric shocks."""
        if seed is not None:
            np.random.seed(seed)

        # Trend (random walk with drift)
        trend = np.cumsum(np.random.normal(0.005, 0.002, T))

        # Asymmetric cycle
        cycle = np.zeros(T)
        for t in range(1, T):
            # Normal small fluctuations
            shock = np.random.normal(0, 0.005)
            # Occasional large negative shocks (recessions)
            if np.random.random() < 0.03:
                shock -= np.abs(np.random.normal(0.04, 0.02))
            cycle[t] = 0.85 * cycle[t-1] + shock
            # Asymmetric recovery - can't go too far above trend
            cycle[t] = min(cycle[t], 0.02)

        return trend + cycle, trend, cycle

    asymm_results = []
    for func, name in filters:
        result = trend_accuracy_test(func, name, plucking_dgp, 'Plucking', n_sims=n_sims)
        asymm_results.append(result)
        print(f"\n{name}:")
        print(f"  Trend RMSE:    {result['trend_rmse']:.4f}")
        print(f"  Cycle RMSE:    {result['cycle_rmse']:.4f}")
        print(f"  Cycle Corr:    {result['cycle_corr']:.3f}")

    asymm_df = pd.DataFrame(asymm_results)
    asymm_df.to_csv(os.path.join(output_dir, 'hamilton_asymmetric_test.csv'), index=False)

    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Spurious cycle magnitude
    ax = axes[0]
    filter_names = [r['filter'] for r in spurious_results]
    spurious_std = [r['mean_cycle_std'] for r in spurious_results]
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    bars = ax.bar(filter_names, spurious_std, color=colors)
    ax.set_ylabel('Spurious Cycle Std Dev')
    ax.set_title('Spurious Cycles in Random Walk\n(Lower = Better)')
    ax.axhline(0, color='black', linewidth=0.5)

    # Panel 2: UC Model Accuracy
    ax = axes[1]
    cycle_corr = [r['cycle_corr'] for r in accuracy_results]
    bars = ax.bar(filter_names, cycle_corr, color=colors)
    ax.set_ylabel('Correlation with True Cycle')
    ax.set_title('UC Model: Cycle Recovery\n(Higher = Better)')
    ax.set_ylim(0, 1)

    # Panel 3: Plucking Model Accuracy
    ax = axes[2]
    cycle_corr_asymm = [r['cycle_corr'] for r in asymm_results]
    bars = ax.bar(filter_names, cycle_corr_asymm, color=colors)
    ax.set_ylabel('Correlation with True Cycle')
    ax.set_title('Plucking Model: Cycle Recovery\n(Higher = Better)')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hamilton_experiments_summary.png'), dpi=300)
    plt.close()

    print(f"\n\nResults saved to {output_dir}")

    return {
        'spurious': spurious_df,
        'accuracy': accuracy_df,
        'asymmetric': asymm_df
    }


if __name__ == '__main__':
    results = run_hamilton_experiments(n_sims=100)  # Reduced for quick test
