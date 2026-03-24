"""
Real-Time Analysis of L1-HP vs L2-HP Filter Revisions

This module implements pseudo real-time analysis using ALFRED (Archival FRED) vintage data.
For each historical vintage date, we apply filters using only data available at that time,
then track how estimates are revised as new data arrives.

Key questions addressed:
1. How much do L1 vs L2 end-of-sample gap estimates get revised?
2. Are revisions systematic (biased in one direction)?
3. Which filter has better end-point stability?
"""

import numpy as np
import pandas as pd
from fredapi import Fred
import os
import sys
import json
from datetime import datetime, timedelta
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter


def load_fred_api_key(config_path=None):
    """Load FRED API key from config file."""
    if config_path is None:
        # Look in project root
        config_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'config.json'
        )

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('FRED_API_KEY')

    # Try environment variable
    return os.environ.get('FRED_API_KEY')


class ALFREDDownloader:
    """
    Download historical vintages from ALFRED (Archival FRED).

    ALFRED stores "what the data looked like" at different points in time,
    enabling pseudo real-time analysis.
    """

    def __init__(self, api_key=None):
        if api_key is None:
            api_key = load_fred_api_key()
        if api_key is None:
            raise ValueError("FRED API key not found. Set FRED_API_KEY or provide config.json")

        self.fred = Fred(api_key=api_key)

    def get_vintage_dates(self, series_id='GDPC1'):
        """
        Get all available vintage dates for a series.

        Returns
        -------
        list of datetime
            Available vintage dates (when data was released/revised)
        """
        # Get vintage dates from ALFRED
        vintage_dates = self.fred.get_series_vintage_dates(series_id)
        return vintage_dates

    def get_vintage_data(self, series_id='GDPC1', vintage_date=None):
        """
        Get data as it appeared on a specific vintage date.

        Parameters
        ----------
        series_id : str
            FRED series ID (e.g., 'GDPC1' for Real GDP)
        vintage_date : str or datetime
            The vintage date (data release date)

        Returns
        -------
        pd.Series
            Data as it appeared on the vintage date, indexed by observation date
        """
        if vintage_date is None:
            # Get latest vintage
            return self.fred.get_series(series_id)

        vintage_dt = pd.Timestamp(vintage_date)

        # Get all revisions
        all_data = self.fred.get_series_as_of_date(series_id, vintage_date)

        if all_data is None or len(all_data) == 0:
            return None

        # Filter to revisions available by the vintage date
        # realtime_start <= vintage_date
        all_data['realtime_start'] = pd.to_datetime(all_data['realtime_start'])
        all_data['date'] = pd.to_datetime(all_data['date'])
        all_data = all_data[all_data['realtime_start'] <= vintage_dt]

        # For each observation date, keep the most recent revision
        # (i.e., the value that was available on the vintage date)
        all_data = all_data.sort_values(['date', 'realtime_start'])
        all_data = all_data.groupby('date').last().reset_index()

        # Convert 'value' to numeric
        all_data['value'] = pd.to_numeric(all_data['value'], errors='coerce')

        # Create Series indexed by observation date
        result = pd.Series(
            all_data['value'].values,
            index=all_data['date'],
            name=series_id
        )

        return result.sort_index()

    def download_all_vintages(self, series_id='GDPC1', start_vintage='2000-01-01',
                              end_vintage=None, frequency='Q'):
        """
        Download all vintages of a series within a date range.

        Parameters
        ----------
        series_id : str
            FRED series ID
        start_vintage : str
            First vintage date to download
        end_vintage : str
            Last vintage date (default: today)
        frequency : str
            How often to sample vintages: 'Q' (quarterly), 'M' (monthly), 'A' (annual)

        Returns
        -------
        dict
            {vintage_date: pd.Series} for each vintage
        """
        # Get all vintage dates
        all_vintages = self.get_vintage_dates(series_id)

        # Filter to date range
        start_dt = pd.Timestamp(start_vintage)
        end_dt = pd.Timestamp(end_vintage) if end_vintage else pd.Timestamp.today()

        filtered_vintages = [v for v in all_vintages
                            if start_dt <= pd.Timestamp(v) <= end_dt]

        # Sample at specified frequency
        if frequency == 'Q':
            # Keep ~quarterly vintages (one per quarter)
            sampled = []
            last_quarter = None
            for v in filtered_vintages:
                v_dt = pd.Timestamp(v)
                quarter = (v_dt.year, v_dt.quarter)
                if quarter != last_quarter:
                    sampled.append(v)
                    last_quarter = quarter
            filtered_vintages = sampled
        elif frequency == 'A':
            # Keep ~annual vintages
            sampled = []
            last_year = None
            for v in filtered_vintages:
                v_dt = pd.Timestamp(v)
                if v_dt.year != last_year:
                    sampled.append(v)
                    last_year = v_dt.year
            filtered_vintages = sampled

        print(f"Downloading {len(filtered_vintages)} vintages of {series_id}...")

        vintages_data = {}
        for vintage in tqdm(filtered_vintages, desc="Downloading vintages"):
            try:
                data = self.get_vintage_data(series_id, vintage)
                if data is not None and len(data) > 0:
                    vintages_data[vintage] = data
            except Exception as e:
                print(f"  Warning: Failed to get vintage {vintage}: {e}")
                continue

        return vintages_data


class RealTimeAnalysis:
    """
    Perform pseudo real-time analysis comparing L1-HP and L2-HP filter revisions.
    """

    def __init__(self, lambda_l1=600, lambda_l2=1600):
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def compute_realtime_gaps(self, vintages_data, min_obs=40):
        """
        For each vintage, compute end-of-sample output gap using L1 and L2 filters.

        Parameters
        ----------
        vintages_data : dict
            {vintage_date: pd.Series} from ALFREDDownloader
        min_obs : int
            Minimum observations required to apply filter

        Returns
        -------
        pd.DataFrame
            Columns: vintage_date, last_obs_date, gap_l1, gap_l2, n_obs
        """
        results = []

        for vintage_date, data in tqdm(vintages_data.items(), desc="Computing gaps"):
            # Clean data
            data = data.dropna()

            if len(data) < min_obs:
                continue

            # Convert to log levels
            y = np.log(data.values)

            try:
                # L2-HP filter
                trend_l2, cycle_l2 = l2_hp_filter(y, lamb=self.lambda_l2)
                gap_l2 = cycle_l2[-1] * 100  # End-of-sample gap in percent

                # L1-HP filter
                trend_l1, cycle_l1 = l1_hp_filter(y, lamb=self.lambda_l1)
                gap_l1 = cycle_l1[-1] * 100

                results.append({
                    'vintage_date': pd.Timestamp(vintage_date),
                    'last_obs_date': data.index[-1],
                    'gap_l1': gap_l1,
                    'gap_l2': gap_l2,
                    'n_obs': len(data)
                })
            except Exception as e:
                print(f"  Warning: Filter failed for vintage {vintage_date}: {e}")
                continue

        return pd.DataFrame(results)

    def compute_revisions(self, vintages_data, min_obs=40, revision_horizon_years=3):
        """
        Compute revisions by tracking the gap at a *fixed* observation date
        across vintages.

        For each observation date d:
        1. Real-time estimate = gap at d from the first vintage containing d
        2. Final estimate = gap at d from a vintage 3+ years later
        3. Revision = final - real-time

        This correctly measures how the filter's estimate of the gap at date d
        changes as more data arrives (unlike the old method which compared
        end-of-sample gaps from different vintages / different dates).

        Parameters
        ----------
        vintages_data : dict
            {vintage_date: pd.Series} from ALFREDDownloader
        min_obs : int
            Minimum observations required to apply filter
        revision_horizon_years : int
            How many years later the "final" vintage should be

        Returns
        -------
        pd.DataFrame
            Columns: obs_date, realtime_gap_l1, realtime_gap_l2,
                     final_gap_l1, final_gap_l2, revision_l1, revision_l2,
                     realtime_vintage, final_vintage
        """
        # Step 1: Build vintage × obs_date gap matrices
        sorted_vintages = sorted(vintages_data.keys())
        gap_records = []  # list of (vintage_date, obs_date_index_series, gap_l1_array, gap_l2_array)

        for vintage in tqdm(sorted_vintages, desc="Building gap matrix"):
            data = vintages_data[vintage].dropna()
            if len(data) < min_obs:
                continue
            y = np.log(data.values)
            try:
                _, cycle_l2 = l2_hp_filter(y, lamb=self.lambda_l2)
                _, cycle_l1 = l1_hp_filter(y, lamb=self.lambda_l1)
                gap_records.append({
                    'vintage': pd.Timestamp(vintage),
                    'obs_dates': data.index,
                    'gap_l1': cycle_l1 * 100,
                    'gap_l2': cycle_l2 * 100,
                })
            except Exception:
                continue

        if len(gap_records) == 0:
            return pd.DataFrame()

        # Step 2: For each observation date, find real-time and final estimates
        results = []
        # Collect all observation dates that appear in at least one vintage
        all_obs_dates = set()
        for rec in gap_records:
            all_obs_dates.update(rec['obs_dates'])
        all_obs_dates = sorted(all_obs_dates)

        for obs_date in all_obs_dates:
            # Real-time: first vintage containing obs_date
            rt_rec = None
            rt_idx = None
            for rec in gap_records:
                if obs_date in rec['obs_dates']:
                    idx = rec['obs_dates'].get_loc(obs_date)
                    rt_rec = rec
                    rt_idx = idx
                    break

            if rt_rec is None:
                continue

            # Final: first vintage at least revision_horizon_years after real-time vintage
            final_cutoff = rt_rec['vintage'] + pd.DateOffset(years=revision_horizon_years)
            final_rec = None
            final_idx = None
            for rec in gap_records:
                if rec['vintage'] >= final_cutoff and obs_date in rec['obs_dates']:
                    final_idx = rec['obs_dates'].get_loc(obs_date)
                    final_rec = rec
                    break

            if final_rec is None:
                continue

            results.append({
                'obs_date': obs_date,
                'realtime_gap_l1': rt_rec['gap_l1'][rt_idx],
                'realtime_gap_l2': rt_rec['gap_l2'][rt_idx],
                'final_gap_l1': final_rec['gap_l1'][final_idx],
                'final_gap_l2': final_rec['gap_l2'][final_idx],
                'revision_l1': final_rec['gap_l1'][final_idx] - rt_rec['gap_l1'][rt_idx],
                'revision_l2': final_rec['gap_l2'][final_idx] - rt_rec['gap_l2'][rt_idx],
                'realtime_vintage': rt_rec['vintage'],
                'final_vintage': final_rec['vintage'],
            })

        return pd.DataFrame(results)

    def compute_revision_statistics(self, realtime_df):
        """
        Compute summary statistics on filter revisions.

        Compares consecutive vintages to measure revision volatility.

        Parameters
        ----------
        realtime_df : pd.DataFrame
            Output from compute_realtime_gaps()

        Returns
        -------
        dict
            Summary statistics for L1 and L2 revisions
        """
        # Sort by vintage date
        df = realtime_df.sort_values('vintage_date').reset_index(drop=True)

        # Compute revisions between consecutive vintages
        df['revision_l1'] = df['gap_l1'].diff()
        df['revision_l2'] = df['gap_l2'].diff()

        # Remove first row (no revision)
        df = df.dropna(subset=['revision_l1', 'revision_l2'])

        stats = {
            'L1-HP': {
                'mean_revision': df['revision_l1'].mean(),
                'std_revision': df['revision_l1'].std(),
                'mae_revision': df['revision_l1'].abs().mean(),
                'max_revision': df['revision_l1'].abs().max(),
                'n_positive': (df['revision_l1'] > 0).sum(),
                'n_negative': (df['revision_l1'] < 0).sum(),
            },
            'L2-HP': {
                'mean_revision': df['revision_l2'].mean(),
                'std_revision': df['revision_l2'].std(),
                'mae_revision': df['revision_l2'].abs().mean(),
                'max_revision': df['revision_l2'].abs().max(),
                'n_positive': (df['revision_l2'] > 0).sum(),
                'n_negative': (df['revision_l2'] < 0).sum(),
            }
        }

        # Add correlation between L1 and L2 revisions
        stats['revision_correlation'] = df['revision_l1'].corr(df['revision_l2'])

        return stats, df


def run_realtime_analysis(output_dir=None, start_vintage='2000-01-01',
                          lambda_l1=600, lambda_l2=1600):
    """
    Run complete real-time analysis pipeline.

    Parameters
    ----------
    output_dir : str
        Directory to save results
    start_vintage : str
        First vintage to download
    lambda_l1 : float
        L1-HP smoothing parameter
    lambda_l2 : float
        L2-HP smoothing parameter

    Returns
    -------
    dict
        Results including DataFrames and statistics
    """
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
    results_dir = os.path.join(output_dir, 'results')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "="*70)
    print("REAL-TIME ANALYSIS: L1-HP vs L2-HP Filter Revisions")
    print("="*70)

    # Step 1: Download vintage data
    print("\n[Step 1] Downloading ALFRED vintage data...")
    downloader = ALFREDDownloader()
    vintages_data = downloader.download_all_vintages(
        series_id='GDPC1',
        start_vintage=start_vintage,
        frequency='Q'  # Quarterly vintages
    )
    print(f"  Downloaded {len(vintages_data)} vintages")

    # Step 2: Compute real-time gaps
    print("\n[Step 2] Computing real-time output gaps...")
    analyzer = RealTimeAnalysis(lambda_l1=lambda_l1, lambda_l2=lambda_l2)
    realtime_df = analyzer.compute_realtime_gaps(vintages_data)
    print(f"  Computed gaps for {len(realtime_df)} vintages")

    # Step 3: Compute revisions (fixed-date methodology)
    print("\n[Step 3] Computing revisions (fixed-date methodology)...")
    revisions_df = analyzer.compute_revisions(vintages_data)

    # Step 3b: Compute revision statistics from consecutive vintages (backward compat)
    print("[Step 3b] Computing consecutive-vintage revision statistics...")
    revision_stats, revisions_consec_df = analyzer.compute_revision_statistics(realtime_df)

    # Print summary
    print("\n" + "="*70)
    print("REVISION STATISTICS SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<25} {'L1-HP':>12} {'L2-HP':>12}")
    print("-"*50)
    print(f"{'Mean revision (pp)':<25} {revision_stats['L1-HP']['mean_revision']:>12.3f} {revision_stats['L2-HP']['mean_revision']:>12.3f}")
    print(f"{'Std deviation (pp)':<25} {revision_stats['L1-HP']['std_revision']:>12.3f} {revision_stats['L2-HP']['std_revision']:>12.3f}")
    print(f"{'Mean absolute (pp)':<25} {revision_stats['L1-HP']['mae_revision']:>12.3f} {revision_stats['L2-HP']['mae_revision']:>12.3f}")
    print(f"{'Max absolute (pp)':<25} {revision_stats['L1-HP']['max_revision']:>12.3f} {revision_stats['L2-HP']['max_revision']:>12.3f}")
    print(f"{'N positive revisions':<25} {revision_stats['L1-HP']['n_positive']:>12d} {revision_stats['L2-HP']['n_positive']:>12d}")
    print(f"{'N negative revisions':<25} {revision_stats['L1-HP']['n_negative']:>12d} {revision_stats['L2-HP']['n_negative']:>12d}")
    print("-"*50)
    print(f"{'L1-L2 revision corr':<25} {revision_stats['revision_correlation']:>12.3f}")

    # Step 4: Generate figures
    print("\n[Step 4] Generating figures...")
    _plot_realtime_results(realtime_df, revisions_df, figures_dir)

    # Step 5: Save results
    print("\n[Step 5] Saving results...")
    realtime_df.to_csv(os.path.join(results_dir, 'realtime_gaps.csv'), index=False)
    revisions_df.to_csv(os.path.join(results_dir, 'realtime_revisions.csv'), index=False)

    # Save summary stats
    stats_df = pd.DataFrame({
        'Metric': ['Mean revision', 'Std revision', 'MAE revision', 'Max revision'],
        'L1-HP': [
            revision_stats['L1-HP']['mean_revision'],
            revision_stats['L1-HP']['std_revision'],
            revision_stats['L1-HP']['mae_revision'],
            revision_stats['L1-HP']['max_revision']
        ],
        'L2-HP': [
            revision_stats['L2-HP']['mean_revision'],
            revision_stats['L2-HP']['std_revision'],
            revision_stats['L2-HP']['mae_revision'],
            revision_stats['L2-HP']['max_revision']
        ]
    })
    stats_df.to_csv(os.path.join(results_dir, 'realtime_revision_stats.csv'), index=False)

    print(f"\nResults saved to {results_dir}")

    return {
        'realtime_gaps': realtime_df,
        'revisions': revisions_df,
        'statistics': revision_stats,
        'vintages_data': vintages_data
    }


def _plot_realtime_results(realtime_df, revisions_df, figures_dir):
    """Generate figures for real-time analysis."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Real-time gaps over time
    ax1 = axes[0, 0]
    ax1.plot(realtime_df['vintage_date'], realtime_df['gap_l2'],
             'b-', alpha=0.7, label='L2-HP', linewidth=1)
    ax1.plot(realtime_df['vintage_date'], realtime_df['gap_l1'],
             'r-', alpha=0.7, label='L1-HP', linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Vintage Date')
    ax1.set_ylabel('End-of-Sample Gap (%)')
    ax1.set_title('Real-Time Output Gap Estimates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Revisions over time
    ax2 = axes[0, 1]
    rev_date_col = 'vintage_date' if 'vintage_date' in revisions_df.columns else 'obs_date'
    ax2.plot(revisions_df[rev_date_col], revisions_df['revision_l2'],
             'b-', alpha=0.7, label='L2-HP', linewidth=1)
    ax2.plot(revisions_df[rev_date_col], revisions_df['revision_l1'],
             'r-', alpha=0.7, label='L1-HP', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Vintage Date')
    ax2.set_ylabel('Revision (pp)')
    ax2.set_title('Gap Revisions Between Vintages')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Distribution of revisions
    ax3 = axes[1, 0]
    ax3.hist(revisions_df['revision_l2'].dropna(), bins=30, alpha=0.5,
             label='L2-HP', color='blue', density=True)
    ax3.hist(revisions_df['revision_l1'].dropna(), bins=30, alpha=0.5,
             label='L1-HP', color='red', density=True)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Revision Size (pp)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distribution of Revisions')
    ax3.legend()

    # Panel 4: L1 vs L2 revision scatter
    ax4 = axes[1, 1]
    ax4.scatter(revisions_df['revision_l2'], revisions_df['revision_l1'],
                alpha=0.5, s=20)

    # Add 45-degree line
    lims = [
        min(ax4.get_xlim()[0], ax4.get_ylim()[0]),
        max(ax4.get_xlim()[1], ax4.get_ylim()[1])
    ]
    ax4.plot(lims, lims, 'k--', alpha=0.5, label='45° line')
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)

    ax4.set_xlabel('L2-HP Revision (pp)')
    ax4.set_ylabel('L1-HP Revision (pp)')
    ax4.set_title('L1 vs L2 Revisions')
    ax4.grid(True, alpha=0.3)

    # Compute and display correlation
    corr = revisions_df['revision_l1'].corr(revisions_df['revision_l2'])
    ax4.text(0.05, 0.95, f'Corr: {corr:.2f}', transform=ax4.transAxes,
             fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'realtime_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir}/realtime_analysis.png")


def run_endpoint_stability_analysis(vintages_data, output_dir=None,
                                    lambda_l1=600, lambda_l2=1600):
    """
    Analyze end-point stability specifically.

    For a fixed historical period, compare how the end-point gap estimate
    changes across vintages.

    Parameters
    ----------
    vintages_data : dict
        Vintage data from ALFREDDownloader
    output_dir : str
        Output directory
    lambda_l1, lambda_l2 : float
        Filter parameters

    Returns
    -------
    pd.DataFrame
        End-point estimates across vintages for specific dates
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
    results_dir = os.path.join(output_dir, 'results')
    figures_dir = os.path.join(output_dir, 'figures')

    print("\n" + "="*70)
    print("END-POINT STABILITY ANALYSIS")
    print("="*70)

    # Key dates to track (recession troughs)
    key_dates = {
        '2001-Q4': pd.Timestamp('2001-12-31'),  # 2001 recession trough
        '2009-Q2': pd.Timestamp('2009-06-30'),  # Financial crisis trough
        '2020-Q2': pd.Timestamp('2020-06-30'),  # COVID trough
    }

    results = []

    # Sort vintages by date
    sorted_vintages = sorted(vintages_data.keys())

    for key_name, key_date in key_dates.items():
        print(f"\n  Tracking {key_name} ({key_date.date()})...")

        for vintage in tqdm(sorted_vintages, desc=f"  {key_name}"):
            vintage_dt = pd.Timestamp(vintage)
            data = vintages_data[vintage]
            data = data.dropna()

            # Check if key_date is in this vintage's data
            if key_date not in data.index:
                # Find closest date
                available_dates = data.index[data.index <= key_date]
                if len(available_dates) == 0:
                    continue
                closest_date = available_dates[-1]
            else:
                closest_date = key_date

            # Get index of the key date
            try:
                idx = data.index.get_loc(closest_date)
            except KeyError:
                continue

            if idx < 40:  # Need minimum data
                continue

            # Filter data up to key date
            y = np.log(data.iloc[:idx+1].values)

            try:
                _, cycle_l2 = l2_hp_filter(y, lamb=lambda_l2)
                _, cycle_l1 = l1_hp_filter(y, lamb=lambda_l1)

                results.append({
                    'key_date': key_name,
                    'vintage_date': vintage_dt,
                    'gap_l2': cycle_l2[-1] * 100,
                    'gap_l1': cycle_l1[-1] * 100,
                    'n_obs': len(y)
                })
            except Exception:
                continue

    df = pd.DataFrame(results)

    # Save and plot
    df.to_csv(os.path.join(results_dir, 'endpoint_stability.csv'), index=False)

    # Plot
    _plot_endpoint_stability(df, figures_dir)

    return df


def _plot_endpoint_stability(df, figures_dir):
    """Plot end-point stability for key recession dates."""
    import matplotlib.pyplot as plt

    key_dates = df['key_date'].unique()
    n_dates = len(key_dates)

    fig, axes = plt.subplots(1, n_dates, figsize=(5*n_dates, 4))
    if n_dates == 1:
        axes = [axes]

    for ax, key_date in zip(axes, key_dates):
        subset = df[df['key_date'] == key_date].sort_values('vintage_date')

        ax.plot(subset['vintage_date'], subset['gap_l2'],
                'b-o', markersize=3, label='L2-HP', alpha=0.7)
        ax.plot(subset['vintage_date'], subset['gap_l1'],
                'r-o', markersize=3, label='L1-HP', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        ax.set_xlabel('Vintage Date')
        ax.set_ylabel('Output Gap (%)')
        ax.set_title(f'Gap Estimate for {key_date}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rotate x labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'endpoint_stability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir}/endpoint_stability.png")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Real-Time Analysis')
    parser.add_argument('--start', default='2000-01-01',
                       help='Start vintage date')
    parser.add_argument('--lambda_l1', type=float, default=600,
                       help='L1-HP lambda')
    parser.add_argument('--lambda_l2', type=float, default=1600,
                       help='L2-HP lambda')
    args = parser.parse_args()

    # Run main analysis
    results = run_realtime_analysis(
        start_vintage=args.start,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2
    )

    # Run endpoint stability analysis
    endpoint_df = run_endpoint_stability_analysis(
        results['vintages_data'],
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2
    )
