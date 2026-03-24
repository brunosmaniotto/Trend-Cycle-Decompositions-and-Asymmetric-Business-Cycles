"""
End-Point Bias Experiments

Replicates and extends Hamilton (2018) end-point instability analysis.
Compares L1 vs L2 filters for end-point stability.

Reference:
- Hamilton (2018) "Why You Should Never Use the Hodrick-Prescott Filter"
- Hodrick (2020) "An Exploration of Trend-Cycle Decomposition Methodologies"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from filters.hp import l2_hp_filter, l1_hp_filter
from filters.hamilton import l2_hamilton_filter, l1_hamilton_filter


def endpoint_revision_analysis(y, filter_func, filter_name, horizons=[4, 8, 12, 16, 20]):
    """
    Analyze how trend/cycle estimates at date t change as sample endpoint moves.

    For each date t (from middle of sample to near end), compute:
    - Estimate using full sample
    - Estimate using sample ending at t, t+4, t+8, etc.
    - Track revision magnitude

    Parameters
    ----------
    y : array-like
        Time series data
    filter_func : callable
        Filter function that returns (trend, cycle)
    filter_name : str
        Name of filter for labeling
    horizons : list
        Additional quarters beyond t to include in "final" estimate

    Returns
    -------
    dict with revision statistics
    """
    T = len(y)
    mid_point = T // 2

    # Get full-sample estimates
    trend_full, cycle_full = filter_func(y)

    # Track revisions for dates in the middle third of sample
    eval_dates = range(mid_point, T - max(horizons) - 1)

    revisions = {h: [] for h in horizons}

    for t in eval_dates:
        # Get estimate at date t using data up to t+h
        for h in horizons:
            if t + h < T:
                y_truncated = y[:t + h + 1]
                trend_trunc, _ = filter_func(y_truncated)
                # Revision = estimate using truncated data - estimate using full sample
                revision = trend_trunc[t] - trend_full[t]
                revisions[h].append(revision)

    # Compute summary statistics
    results = {
        'filter': filter_name,
        'mean_abs_revision': {},
        'max_abs_revision': {},
        'rmse_revision': {}
    }

    for h in horizons:
        if revisions[h]:
            rev_array = np.array(revisions[h])
            results['mean_abs_revision'][h] = np.mean(np.abs(rev_array))
            results['max_abs_revision'][h] = np.max(np.abs(rev_array))
            results['rmse_revision'][h] = np.sqrt(np.mean(rev_array**2))

    return results


def real_time_vs_final_analysis(y, dates, filter_funcs, filter_names, window=40):
    """
    Compare real-time estimates (using only data available at time t)
    with final revised estimates (using full sample).

    Parameters
    ----------
    y : array-like
        Time series data
    dates : array-like
        Date index
    filter_funcs : list
        List of filter functions
    filter_names : list
        Names for each filter
    window : int
        Minimum window for estimation

    Returns
    -------
    DataFrame with real-time and final estimates
    """
    T = len(y)

    results = {'date': dates[window:]}

    for func, name in zip(filter_funcs, filter_names):
        # Full sample estimates
        trend_full, cycle_full = func(y)

        # Real-time estimates
        real_time_trend = np.full(T, np.nan)
        real_time_cycle = np.full(T, np.nan)

        for t in range(window, T):
            trend_t, cycle_t = func(y[:t+1])
            real_time_trend[t] = trend_t[-1]
            real_time_cycle[t] = cycle_t[-1]

        results[f'{name}_trend_final'] = trend_full[window:]
        results[f'{name}_trend_realtime'] = real_time_trend[window:]
        results[f'{name}_cycle_final'] = cycle_full[window:]
        results[f'{name}_cycle_realtime'] = real_time_cycle[window:]
        results[f'{name}_trend_revision'] = trend_full[window:] - real_time_trend[window:]
        results[f'{name}_cycle_revision'] = cycle_full[window:] - real_time_cycle[window:]

    return pd.DataFrame(results)


def run_endpoint_experiments(y, dates=None, output_dir=None, lam=1600):
    """
    Run full suite of endpoint bias experiments.

    Parameters
    ----------
    y : array-like
        Time series (log GDP or similar)
    dates : array-like, optional
        Date index
    output_dir : str, optional
        Directory for output files
    lam : float
        HP filter smoothing parameter
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'experiments')
    os.makedirs(output_dir, exist_ok=True)

    if dates is None:
        dates = np.arange(len(y))

    # Define filters
    filter_funcs = [
        lambda x: l2_hp_filter(x, lamb=lam),
        lambda x: l1_hp_filter(x, lamb=lam),
        lambda x: l2_hamilton_filter(x),
        lambda x: l1_hamilton_filter(x)
    ]
    filter_names = ['HP-L2', 'HP-L1', 'Hamilton-L2', 'Hamilton-L1']

    print("=" * 60)
    print("END-POINT BIAS ANALYSIS")
    print("=" * 60)

    # 1. Revision analysis
    print("\n1. Revision Analysis (how estimates change as sample grows)")
    print("-" * 60)

    all_results = []
    for func, name in zip(filter_funcs, filter_names):
        results = endpoint_revision_analysis(y, func, name)
        all_results.append(results)

        print(f"\n{name}:")
        print(f"  Horizon | Mean Abs Rev | Max Abs Rev | RMSE Rev")
        for h in results['mean_abs_revision'].keys():
            print(f"  {h:7d} | {results['mean_abs_revision'][h]:12.4f} | "
                  f"{results['max_abs_revision'][h]:11.4f} | {results['rmse_revision'][h]:8.4f}")

    # 2. Real-time vs final comparison
    print("\n2. Real-time vs Final Estimates")
    print("-" * 60)

    rt_df = real_time_vs_final_analysis(y, dates, filter_funcs, filter_names)

    for name in filter_names:
        rev = rt_df[f'{name}_cycle_revision'].dropna()
        print(f"\n{name} Cycle Revisions:")
        print(f"  Mean Abs: {np.mean(np.abs(rev)):.4f}")
        print(f"  Max Abs:  {np.max(np.abs(rev)):.4f}")
        print(f"  Std Dev:  {np.std(rev):.4f}")

    # Save results
    rt_df.to_csv(os.path.join(output_dir, 'endpoint_realtime_vs_final.csv'))

    # 3. Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, name in enumerate(filter_names):
        ax = axes[idx // 2, idx % 2]
        rev = rt_df[f'{name}_cycle_revision'].values

        ax.plot(rt_df['date'], rev, alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_title(f'{name}: Cycle Revision (Final - Real-time)')
        ax.set_ylabel('Revision')

        # Add RMSE annotation
        rmse = np.sqrt(np.nanmean(rev**2))
        ax.text(0.02, 0.98, f'RMSE: {rmse:.4f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'endpoint_bias_comparison.png'), dpi=300)
    plt.close()

    print(f"\nResults saved to {output_dir}")

    return all_results, rt_df


def endpoint_bias_by_recession(y, dates, recession_dates, filter_funcs, filter_names):
    """
    Analyze whether endpoint bias is larger during recessions.

    Parameters
    ----------
    y : array-like
        Time series
    dates : array-like
        Date index
    recession_dates : list of tuples
        List of (start, end) dates for recessions
    filter_funcs : list
        Filter functions
    filter_names : list
        Filter names

    Returns
    -------
    DataFrame with bias by recession status
    """
    rt_df = real_time_vs_final_analysis(y, dates, filter_funcs, filter_names)

    # Create recession indicator
    is_recession = np.zeros(len(rt_df), dtype=bool)
    for start, end in recession_dates:
        mask = (rt_df['date'] >= start) & (rt_df['date'] <= end)
        is_recession[mask] = True

    rt_df['recession'] = is_recession

    results = []
    for name in filter_names:
        rev = rt_df[f'{name}_cycle_revision']

        rec_rev = rev[rt_df['recession']].dropna()
        exp_rev = rev[~rt_df['recession']].dropna()

        results.append({
            'filter': name,
            'recession_mean_abs_rev': np.mean(np.abs(rec_rev)),
            'expansion_mean_abs_rev': np.mean(np.abs(exp_rev)),
            'recession_rmse': np.sqrt(np.mean(rec_rev**2)),
            'expansion_rmse': np.sqrt(np.mean(exp_rev**2)),
            'ratio': np.mean(np.abs(rec_rev)) / np.mean(np.abs(exp_rev)) if len(exp_rev) > 0 else np.nan
        })

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Example with simulated data
    np.random.seed(42)
    T = 200

    # Simulate plucking model
    trend = np.cumsum(np.random.normal(0.005, 0.002, T))
    cycle = np.zeros(T)
    for t in range(1, T):
        shock = np.random.normal(0, 0.01)
        if np.random.random() < 0.05:  # Occasional large negative shock
            shock -= 0.05
        cycle[t] = 0.7 * cycle[t-1] + shock

    y = trend + cycle
    dates = pd.date_range('1960-01-01', periods=T, freq='Q')

    results, rt_df = run_endpoint_experiments(y, dates)
