"""
Robustness Check Experiments

Tests filter performance under various conditions:
1. Lambda (λ) sensitivity analysis
2. Different sample periods
3. Subsample stability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

from filters.hp import l2_hp_filter, l1_hp_filter
from filters.hamilton import l2_hamilton_filter, l1_hamilton_filter

try:
    import pandas_datareader.data as web
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False


def lambda_sensitivity(y, lambdas=[100, 400, 800, 1600, 3200, 6400, 12800],
                       filter_type='hp'):
    """
    Analyze sensitivity of filter output to smoothing parameter λ.

    Parameters
    ----------
    y : array-like
        Time series data
    lambdas : list
        λ values to test
    filter_type : str
        'hp' for HP filter

    Returns
    -------
    dict with sensitivity metrics
    """
    results = {'lambda': [], 'filter': [], 'cycle_std': [], 'cycle_skew': [],
               'trend_growth': [], 'cycle_autocorr': []}

    for lam in lambdas:
        # L2 filter
        try:
            trend_l2, cycle_l2 = l2_hp_filter(y, lamb=lam)
            results['lambda'].append(lam)
            results['filter'].append('L2')
            results['cycle_std'].append(np.std(cycle_l2))
            results['cycle_skew'].append(pd.Series(cycle_l2).skew())
            results['trend_growth'].append(np.mean(np.diff(trend_l2)))
            results['cycle_autocorr'].append(np.corrcoef(cycle_l2[:-1], cycle_l2[1:])[0, 1])
        except Exception as e:
            print(f"L2 failed for λ={lam}: {e}")

        # L1 filter
        try:
            trend_l1, cycle_l1 = l1_hp_filter(y, lamb=lam)
            results['lambda'].append(lam)
            results['filter'].append('L1')
            results['cycle_std'].append(np.std(cycle_l1))
            results['cycle_skew'].append(pd.Series(cycle_l1).skew())
            results['trend_growth'].append(np.mean(np.diff(trend_l1)))
            results['cycle_autocorr'].append(np.corrcoef(cycle_l1[:-1], cycle_l1[1:])[0, 1])
        except Exception as e:
            print(f"L1 failed for λ={lam}: {e}")

    return pd.DataFrame(results)


def sample_period_analysis(fred_series='GDPC1', periods=None):
    """
    Compare filter results across different historical periods.

    Parameters
    ----------
    fred_series : str
        FRED series code
    periods : list of tuples
        List of (name, start_date, end_date) tuples

    Returns
    -------
    dict with results by period
    """
    if not HAS_DATAREADER:
        print("pandas_datareader not available. Skipping sample period analysis.")
        return None

    if periods is None:
        periods = [
            ('Full Sample', '1947-01-01', '2024-01-01'),
            ('Pre-Great Moderation', '1947-01-01', '1984-01-01'),
            ('Great Moderation', '1984-01-01', '2007-01-01'),
            ('Post-2008', '2008-01-01', '2024-01-01'),
            ('Pre-COVID', '1947-01-01', '2020-01-01'),
            ('Volcker Era', '1979-01-01', '1987-01-01'),
        ]

    # Download data
    try:
        df = web.DataReader(fred_series, 'fred', '1947-01-01', '2024-12-31')
        y_full = np.log(df[fred_series].values) * 100  # Log and scale
        dates_full = df.index
    except Exception as e:
        print(f"Could not download {fred_series}: {e}")
        return None

    results = []

    for period_name, start, end in periods:
        mask = (dates_full >= start) & (dates_full <= end)
        y = y_full[mask]
        dates = dates_full[mask]

        if len(y) < 40:  # Need minimum sample
            continue

        try:
            # HP-L2
            trend_l2, cycle_l2 = l2_hp_filter(y, lamb=1600)
            # HP-L1
            trend_l1, cycle_l1 = l1_hp_filter(y, lamb=1600)

            # Compute statistics
            results.append({
                'period': period_name,
                'n_obs': len(y),
                'start': start,
                'end': end,
                'l2_cycle_std': np.std(cycle_l2),
                'l1_cycle_std': np.std(cycle_l1),
                'l2_cycle_skew': pd.Series(cycle_l2).skew(),
                'l1_cycle_skew': pd.Series(cycle_l1).skew(),
                'l2_cycle_kurt': pd.Series(cycle_l2).kurtosis(),
                'l1_cycle_kurt': pd.Series(cycle_l1).kurtosis(),
                'cycle_corr': np.corrcoef(cycle_l1, cycle_l2)[0, 1],
                'mean_abs_diff': np.mean(np.abs(cycle_l1 - cycle_l2))
            })

        except Exception as e:
            print(f"Error in period {period_name}: {e}")

    return pd.DataFrame(results)


def subsample_stability(y, dates=None, n_splits=2):
    """
    Test filter stability by comparing estimates from first and second halves.

    Parameters
    ----------
    y : array-like
        Time series
    dates : array-like, optional
        Date index
    n_splits : int
        Number of subsample splits

    Returns
    -------
    DataFrame with stability metrics
    """
    T = len(y)
    split_size = T // n_splits

    results = []

    # Get full-sample estimates
    trend_l2_full, cycle_l2_full = l2_hp_filter(y, lamb=1600)
    trend_l1_full, cycle_l1_full = l1_hp_filter(y, lamb=1600)

    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else T

        y_sub = y[start_idx:end_idx]

        try:
            # HP-L2
            trend_l2, cycle_l2 = l2_hp_filter(y_sub, lamb=1600)
            # HP-L1
            trend_l1, cycle_l1 = l1_hp_filter(y_sub, lamb=1600)

            results.append({
                'subsample': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'n_obs': len(y_sub),
                'l2_cycle_std': np.std(cycle_l2),
                'l1_cycle_std': np.std(cycle_l1),
                'l2_cycle_skew': pd.Series(cycle_l2).skew(),
                'l1_cycle_skew': pd.Series(cycle_l1).skew(),
            })

        except Exception as e:
            print(f"Error in subsample {i+1}: {e}")

    return pd.DataFrame(results)


def agreement_analysis(y, dates=None, threshold=0.5):
    """
    Analyze agreement between L1 and L2 filters.

    When filters agree: likely symmetric dynamics
    When filters disagree: likely asymmetric dynamics

    Parameters
    ----------
    y : array-like
        Time series
    dates : array-like, optional
        Date index
    threshold : float
        Threshold (in std devs) for "significant" disagreement

    Returns
    -------
    dict with agreement analysis
    """
    trend_l2, cycle_l2 = l2_hp_filter(y, lamb=1600)
    trend_l1, cycle_l1 = l1_hp_filter(y, lamb=1600)

    # Compute difference
    diff = cycle_l1 - cycle_l2
    diff_std = np.std(diff)

    # Identify periods of disagreement
    disagreement = np.abs(diff) > threshold * diff_std

    # Compute statistics
    results = {
        'overall_correlation': np.corrcoef(cycle_l1, cycle_l2)[0, 1],
        'mean_abs_difference': np.mean(np.abs(diff)),
        'max_abs_difference': np.max(np.abs(diff)),
        'pct_disagreement': np.mean(disagreement) * 100,
        'l2_skewness': pd.Series(cycle_l2).skew(),
        'l1_skewness': pd.Series(cycle_l1).skew(),
    }

    # If dates provided, identify specific disagreement periods
    if dates is not None:
        disagreement_dates = dates[disagreement]
        results['disagreement_periods'] = list(disagreement_dates)

    return results, diff, disagreement


def run_robustness_experiments(output_dir=None):
    """
    Run full suite of robustness experiments.

    Parameters
    ----------
    output_dir : str, optional
        Output directory
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'experiments')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ROBUSTNESS CHECK EXPERIMENTS")
    print("=" * 70)

    # 1. Lambda Sensitivity (using simulated data first)
    print("\n1. LAMBDA SENSITIVITY ANALYSIS")
    print("-" * 70)

    # Generate test data
    np.random.seed(42)
    T = 200
    trend = np.cumsum(np.random.normal(0.005, 0.002, T))
    cycle = np.zeros(T)
    for t in range(1, T):
        shock = np.random.normal(0, 0.01)
        if np.random.random() < 0.05:
            shock -= 0.04
        cycle[t] = 0.8 * cycle[t-1] + shock
    y_sim = trend + cycle

    lambda_df = lambda_sensitivity(y_sim)
    lambda_df.to_csv(os.path.join(output_dir, 'lambda_sensitivity.csv'), index=False)

    print("\nLambda Sensitivity Results:")
    print(lambda_df.to_string(index=False))

    # Plot lambda sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for filter_type in ['L1', 'L2']:
        df_sub = lambda_df[lambda_df['filter'] == filter_type]

        # Cycle std
        axes[0, 0].plot(df_sub['lambda'], df_sub['cycle_std'],
                        'o-', label=filter_type, linewidth=2)
        # Cycle skewness
        axes[0, 1].plot(df_sub['lambda'], df_sub['cycle_skew'],
                        'o-', label=filter_type, linewidth=2)
        # Trend growth
        axes[1, 0].plot(df_sub['lambda'], df_sub['trend_growth'],
                        'o-', label=filter_type, linewidth=2)
        # Cycle autocorr
        axes[1, 1].plot(df_sub['lambda'], df_sub['cycle_autocorr'],
                        'o-', label=filter_type, linewidth=2)

    axes[0, 0].set_xlabel('λ')
    axes[0, 0].set_ylabel('Cycle Std Dev')
    axes[0, 0].set_title('Cycle Volatility vs λ')
    axes[0, 0].set_xscale('log')
    axes[0, 0].legend()

    axes[0, 1].set_xlabel('λ')
    axes[0, 1].set_ylabel('Cycle Skewness')
    axes[0, 1].set_title('Cycle Skewness vs λ')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()

    axes[1, 0].set_xlabel('λ')
    axes[1, 0].set_ylabel('Mean Trend Growth')
    axes[1, 0].set_title('Trend Growth vs λ')
    axes[1, 0].set_xscale('log')
    axes[1, 0].legend()

    axes[1, 1].set_xlabel('λ')
    axes[1, 1].set_ylabel('Cycle AR(1)')
    axes[1, 1].set_title('Cycle Persistence vs λ')
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lambda_sensitivity.png'), dpi=300)
    plt.close()

    # 2. Sample Period Analysis
    print("\n\n2. SAMPLE PERIOD ANALYSIS")
    print("-" * 70)

    if HAS_DATAREADER:
        period_df = sample_period_analysis()
        if period_df is not None:
            period_df.to_csv(os.path.join(output_dir, 'sample_period_analysis.csv'), index=False)
            print("\nSample Period Results:")
            print(period_df[['period', 'n_obs', 'l2_cycle_skew', 'l1_cycle_skew',
                            'cycle_corr', 'mean_abs_diff']].to_string(index=False))
    else:
        print("pandas_datareader not available. Using simulated data.")
        period_df = None

    # 3. Subsample Stability
    print("\n\n3. SUBSAMPLE STABILITY ANALYSIS")
    print("-" * 70)

    stability_df = subsample_stability(y_sim)
    stability_df.to_csv(os.path.join(output_dir, 'subsample_stability.csv'), index=False)
    print("\nSubsample Stability Results:")
    print(stability_df.to_string(index=False))

    # 4. Agreement Analysis
    print("\n\n4. L1-L2 AGREEMENT ANALYSIS")
    print("-" * 70)

    agreement_results, diff, disagreement = agreement_analysis(y_sim)

    print(f"\nOverall Correlation: {agreement_results['overall_correlation']:.3f}")
    print(f"Mean Absolute Difference: {agreement_results['mean_abs_difference']:.4f}")
    print(f"Percent Disagreement: {agreement_results['pct_disagreement']:.1f}%")
    print(f"L2 Skewness: {agreement_results['l2_skewness']:.3f}")
    print(f"L1 Skewness: {agreement_results['l1_skewness']:.3f}")

    # Save agreement results
    with open(os.path.join(output_dir, 'agreement_analysis.txt'), 'w') as f:
        for key, val in agreement_results.items():
            if key != 'disagreement_periods':
                f.write(f"{key}: {val}\n")

    print(f"\n\nResults saved to {output_dir}")

    return {
        'lambda_sensitivity': lambda_df,
        'sample_periods': period_df,
        'subsample_stability': stability_df,
        'agreement': agreement_results
    }


if __name__ == '__main__':
    results = run_robustness_experiments()
