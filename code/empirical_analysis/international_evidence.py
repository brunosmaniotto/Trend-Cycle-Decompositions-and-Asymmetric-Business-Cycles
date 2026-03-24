"""
International Evidence Experiments

Applies L1 vs L2 filters to GDP data from major economies:
- United States
- United Kingdom
- Germany
- Japan
- France
- Canada

Tests whether asymmetric business cycles are a universal phenomenon.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

from filters.hp import l2_hp_filter, l1_hp_filter
from filters.hamilton import l2_hamilton_filter, l1_hamilton_filter

try:
    import pandas_datareader.data as web
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False


# FRED series codes for international GDP
COUNTRY_SERIES = {
    'United States': 'GDPC1',       # Real GDP, Billions of Chained 2017 Dollars
    'United Kingdom': 'CLVMNACSCAB1GQUK',  # Real GDP, UK
    'Germany': 'CLVMNACSCAB1GQDE',  # Real GDP, Germany
    'Japan': 'JPNRGDPEXP',          # Real GDP, Japan
    'France': 'CLVMNACSCAB1GQFR',   # Real GDP, France
    'Canada': 'NGDPRSAXDCCAQ',      # Real GDP, Canada
    'Euro Area': 'CLVMNACSCAB1GQEA19',  # Real GDP, Euro Area
    'Australia': 'AUSGDPDEFQISMEI',  # Real GDP, Australia
}

# Alternative OECD-style codes if FRED codes don't work
BACKUP_SERIES = {
    'United Kingdom': 'UKNGDP',
    'Germany': 'DEUNGDP',
    'Japan': 'JPNNGDP',
}


def download_international_gdp(start_date='1960-01-01', end_date='2024-01-01'):
    """
    Download GDP data for multiple countries from FRED.

    Returns
    -------
    dict of DataFrames, keyed by country name
    """
    if not HAS_DATAREADER:
        print("pandas_datareader not available. Cannot download data.")
        return None

    data = {}

    for country, series in COUNTRY_SERIES.items():
        try:
            df = web.DataReader(series, 'fred', start_date, end_date)
            # Convert to log scale and multiply by 100 for percentage interpretation
            df['log_gdp'] = np.log(df[series]) * 100
            data[country] = df
            print(f"Downloaded {country}: {len(df)} observations")
        except Exception as e:
            print(f"Failed to download {country} ({series}): {e}")
            # Try backup series
            if country in BACKUP_SERIES:
                try:
                    backup = BACKUP_SERIES[country]
                    df = web.DataReader(backup, 'fred', start_date, end_date)
                    df['log_gdp'] = np.log(df[backup]) * 100
                    data[country] = df
                    print(f"  -> Used backup series {backup}")
                except Exception as e2:
                    print(f"  -> Backup also failed: {e2}")

    return data


def analyze_country(country_name, y, dates=None, lam_l2=1600, lam_l1=600):
    """
    Apply all filters to a single country's GDP data and compute statistics.

    Parameters
    ----------
    country_name : str
        Country name
    y : array-like
        Log GDP (×100)
    dates : array-like, optional
        Date index
    lam_l2 : float
        HP filter smoothing parameter for L2 (standard = 1600)
    lam_l1 : float
        HP filter smoothing parameter for L1 (robust = 600)

    Returns
    -------
    dict with results
    """
    results = {'country': country_name, 'n_obs': len(y)}

    # Apply filters
    try:
        trend_l2, cycle_l2 = l2_hp_filter(y, lamb=lam_l2)
        trend_l1, cycle_l1 = l1_hp_filter(y, lamb=lam_l1)

        # Cycle statistics - L2
        results['l2_cycle_std'] = np.std(cycle_l2)
        results['l2_cycle_skew'] = stats.skew(cycle_l2)
        results['l2_cycle_kurt'] = stats.kurtosis(cycle_l2)
        results['l2_cycle_min'] = np.min(cycle_l2)
        results['l2_cycle_max'] = np.max(cycle_l2)

        # Cycle statistics - L1
        results['l1_cycle_std'] = np.std(cycle_l1)
        results['l1_cycle_skew'] = stats.skew(cycle_l1)
        results['l1_cycle_kurt'] = stats.kurtosis(cycle_l1)
        results['l1_cycle_min'] = np.min(cycle_l1)
        results['l1_cycle_max'] = np.max(cycle_l1)

        # Comparison
        results['cycle_corr'] = np.corrcoef(cycle_l1, cycle_l2)[0, 1]
        results['mean_abs_diff'] = np.mean(np.abs(cycle_l1 - cycle_l2))

        # Asymmetry test (Pearson's test for skewness significance)
        n = len(cycle_l2)
        se_skew = np.sqrt(6 / n)  # Standard error of skewness
        results['l2_skew_tstat'] = results['l2_cycle_skew'] / se_skew
        results['l1_skew_tstat'] = results['l1_cycle_skew'] / se_skew

        # Store cycles for plotting
        results['cycle_l2'] = cycle_l2
        results['cycle_l1'] = cycle_l1
        results['dates'] = dates

    except Exception as e:
        print(f"Error analyzing {country_name}: {e}")
        return None

    return results


def run_international_experiments(output_dir=None, lam_l2=1600, lam_l1=600):
    """
    Run international evidence experiments.

    Parameters
    ----------
    output_dir : str, optional
        Output directory
    lam_l2 : float
        HP filter smoothing parameter for L2 (standard)
    lam_l1 : float
        HP filter smoothing parameter for L1 (robust)
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'experiments')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("INTERNATIONAL EVIDENCE EXPERIMENTS")
    print("=" * 70)

    # Download data
    print("\nDownloading international GDP data...")
    data = download_international_gdp()

    if data is None or len(data) == 0:
        print("No data available. Exiting.")
        return None

    # Analyze each country
    print("\n\nAnalyzing countries...")
    print("-" * 70)

    all_results = []
    country_cycles = {}

    for country, df in data.items():
        y = df['log_gdp'].dropna().values
        dates = df.index[df['log_gdp'].notna()]

        if len(y) < 40:
            print(f"Skipping {country}: insufficient data ({len(y)} obs)")
            continue

        result = analyze_country(country, y, dates, lam_l2=lam_l2, lam_l1=lam_l1)

        if result is not None:
            country_cycles[country] = {
                'dates': result.pop('dates'),
                'cycle_l2': result.pop('cycle_l2'),
                'cycle_l1': result.pop('cycle_l1')
            }
            all_results.append(result)

            print(f"\n{country}:")
            print(f"  N obs: {result['n_obs']}")
            print(f"  L2 Cycle Skewness: {result['l2_cycle_skew']:6.3f} (t={result['l2_skew_tstat']:.2f})")
            print(f"  L1 Cycle Skewness: {result['l1_cycle_skew']:6.3f} (t={result['l1_skew_tstat']:.2f})")
            print(f"  L1-L2 Correlation: {result['cycle_corr']:.3f}")

    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'international_evidence.csv'), index=False)

    # Summary statistics
    print("\n\n" + "=" * 70)
    print("SUMMARY: ASYMMETRY ACROSS COUNTRIES")
    print("=" * 70)

    print(f"\nCountries with significant negative L2 skewness (t < -1.96):")
    sig_neg_l2 = results_df[results_df['l2_skew_tstat'] < -1.96]['country'].tolist()
    print(f"  {', '.join(sig_neg_l2) if sig_neg_l2 else 'None'}")

    print(f"\nCountries with significant negative L1 skewness (t < -1.96):")
    sig_neg_l1 = results_df[results_df['l1_skew_tstat'] < -1.96]['country'].tolist()
    print(f"  {', '.join(sig_neg_l1) if sig_neg_l1 else 'None'}")

    print(f"\nAverage L2 skewness: {results_df['l2_cycle_skew'].mean():.3f}")
    print(f"Average L1 skewness: {results_df['l1_cycle_skew'].mean():.3f}")
    print(f"Average L1-L2 correlation: {results_df['cycle_corr'].mean():.3f}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Skewness comparison bar chart
    ax = axes[0, 0]
    countries = results_df['country'].values
    x = np.arange(len(countries))
    width = 0.35

    ax.bar(x - width/2, results_df['l2_cycle_skew'], width, label='L2', color='#d62728', alpha=0.7)
    ax.bar(x + width/2, results_df['l1_cycle_skew'], width, label='L1', color='#2ca02c', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    n_median = int(np.median(results_df['n_obs'])) if 'n_obs' in results_df.columns else 180
    ax.axhline(-1.96 * np.sqrt(6 / n_median), color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha='right')
    ax.set_ylabel('Cycle Skewness')
    ax.set_title('Business Cycle Skewness by Country')
    ax.legend()

    # Panel 2: L1-L2 Correlation
    ax = axes[0, 1]
    ax.bar(countries, results_df['cycle_corr'], color='#1f77b4', alpha=0.7)
    ax.axhline(1.0, color='black', linewidth=0.5)
    ax.set_xticklabels(countries, rotation=45, ha='right')
    ax.set_ylabel('L1-L2 Cycle Correlation')
    ax.set_title('Agreement Between L1 and L2 Filters')
    ax.set_ylim(0.5, 1.0)

    # Panel 3: Cycle volatility
    ax = axes[1, 0]
    ax.bar(x - width/2, results_df['l2_cycle_std'], width, label='L2', color='#d62728', alpha=0.7)
    ax.bar(x + width/2, results_df['l1_cycle_std'], width, label='L1', color='#2ca02c', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=45, ha='right')
    ax.set_ylabel('Cycle Std Dev (%)')
    ax.set_title('Business Cycle Volatility by Country')
    ax.legend()

    # Panel 4: Example - US cycles
    ax = axes[1, 1]
    if 'United States' in country_cycles:
        us = country_cycles['United States']
        ax.plot(us['dates'], us['cycle_l2'], label='L2', alpha=0.7, color='#d62728')
        ax.plot(us['dates'], us['cycle_l1'], label='L1', alpha=0.7, color='#2ca02c')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_ylabel('Cycle (%)')
        ax.set_title('United States: L1 vs L2 Cycles')
        ax.legend()

        # Shade recessions (approximate)
        recession_periods = [
            ('1973-11-01', '1975-03-01'),
            ('1980-01-01', '1980-07-01'),
            ('1981-07-01', '1982-11-01'),
            ('1990-07-01', '1991-03-01'),
            ('2001-03-01', '2001-11-01'),
            ('2007-12-01', '2009-06-01'),
            ('2020-02-01', '2020-04-01'),
        ]
        for start, end in recession_periods:
            try:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                          alpha=0.2, color='gray')
            except Exception:
                pass

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'international_evidence.png'), dpi=300)
    plt.close()

    # Create individual country cycle plots
    if len(country_cycles) > 1:
        n_countries = len(country_cycles)
        n_cols = 2
        n_rows = (n_countries + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten() if n_countries > 2 else [axes] if n_countries == 1 else axes

        for idx, (country, cycles) in enumerate(country_cycles.items()):
            ax = axes[idx]
            ax.plot(cycles['dates'], cycles['cycle_l2'], label='L2', alpha=0.7, color='#d62728')
            ax.plot(cycles['dates'], cycles['cycle_l1'], label='L1', alpha=0.7, color='#2ca02c')
            ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax.set_title(country)
            ax.legend(loc='upper right')
            ax.set_ylabel('Cycle (%)')

        # Hide unused subplots
        for idx in range(len(country_cycles), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'international_cycles_all.png'), dpi=300)
        plt.close()

    print(f"\n\nResults saved to {output_dir}")

    return results_df


if __name__ == '__main__':
    results = run_international_experiments()
