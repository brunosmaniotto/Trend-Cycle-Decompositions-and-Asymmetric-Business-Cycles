"""
CBO Output Gap Comparison

Compares output gap estimates from:
1. CBO (production function approach) - GDPPOT from FRED
2. L2-HP filter (standard)
3. L1-HP filter (robust)

Key question: Does the L1 filter align more with a "plucking" view (potential stays high)
while CBO/L2-HP show potential falling during recessions (hysteresis view)?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas_datareader as pdr
from scipy import stats
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter


def download_cbo_data(start_date='1949-01-01', end_date='2024-12-31'):
    """
    Download Real GDP and CBO Potential GDP from FRED.
    """
    print("Downloading GDP and CBO Potential Output from FRED...")

    # GDPC1: Real GDP (Billions of Chained 2017 Dollars, Quarterly)
    # GDPPOT: Real Potential GDP (Billions of Chained 2017 Dollars, Quarterly)
    gdp = pdr.get_data_fred('GDPC1', start=start_date, end=end_date)
    gdppot = pdr.get_data_fred('GDPPOT', start=start_date, end=end_date)
    recession = pdr.get_data_fred('USREC', start=start_date, end=end_date)

    # Merge
    df = pd.concat([gdp, gdppot, recession], axis=1)
    df.columns = ['GDP', 'GDP_Potential_CBO', 'Recession']

    # Resample to quarterly (GDPPOT is quarterly, GDP is quarterly)
    df = df.resample('QE').last().dropna()

    print(f"  Downloaded {len(df)} quarterly observations")
    print(f"  Date range: {_quarter_label(df.index[0])} to {_quarter_label(df.index[-1])}")

    return df


def compute_output_gaps(df, l1_lambda=600, l2_lambda=1600):
    """
    Compute output gaps using CBO, L2-HP, and L1-HP methods.

    Output gap = (Y - Y*) / Y* * 100  (percentage deviation from potential)
    """
    # Log GDP for filtering
    gdp_log = np.log(df['GDP'].values) * 100  # Log * 100 for percentage interpretation

    # CBO output gap (already in levels)
    df['Gap_CBO'] = (df['GDP'] - df['GDP_Potential_CBO']) / df['GDP_Potential_CBO'] * 100

    # L2-HP filter
    trend_l2, cycle_l2 = l2_hp_filter(gdp_log, lamb=l2_lambda)
    df['Trend_L2'] = trend_l2
    df['Gap_L2'] = cycle_l2  # Already in percentage points (since we used log*100)

    # L1-HP filter
    trend_l1, cycle_l1 = l1_hp_filter(gdp_log, lamb=l1_lambda)
    df['Trend_L1'] = trend_l1
    df['Gap_L1'] = cycle_l1

    return df


def compute_gap_statistics(df):
    """
    Compute summary statistics for each output gap measure.
    """
    gap_cols = ['Gap_CBO', 'Gap_L2', 'Gap_L1']

    stats_list = []
    for col in gap_cols:
        gap = df[col].dropna()
        stats_list.append({
            'Measure': col.replace('Gap_', ''),
            'Mean': gap.mean(),
            'Std': gap.std(),
            'Min': gap.min(),
            'Max': gap.max(),
            'Skewness': stats.skew(gap),
            'Kurtosis': stats.kurtosis(gap)
        })

    return pd.DataFrame(stats_list)


def compute_recession_gaps(df):
    """
    Compute average output gap during NBER recessions for each measure.
    """
    # Identify recession quarters
    recession_mask = df['Recession'] == 1

    results = []

    # Overall recession statistics
    for col in ['Gap_CBO', 'Gap_L2', 'Gap_L1']:
        gap_recession = df.loc[recession_mask, col].dropna()
        gap_expansion = df.loc[~recession_mask, col].dropna()

        results.append({
            'Measure': col.replace('Gap_', ''),
            'Mean_Recession': gap_recession.mean(),
            'Mean_Expansion': gap_expansion.mean(),
            'Min_Recession': gap_recession.min(),
            'Asymmetry': gap_recession.mean() - gap_expansion.mean()
        })

    return pd.DataFrame(results)


def _quarter_label(dt):
    """Format a datetime as 'YYYY-QN'."""
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"


def compute_specific_recession_gaps(df):
    """
    Compute trough and peak output gaps for each post-1960 NBER recession.

    Trough is the quarter with the minimum CBO output gap within the NBER
    recession window or up to two quarters after the NBER trough date.  All
    three gap measures (CBO, L1, L2) are reported at this same quarter.

    Pre-recession peak gap is the output gap in the quarter immediately
    before the NBER peak.

    Length is the number of quarters from NBER peak to CBO trough.
    Recovery is the number of quarters from trough until the CBO gap
    returns to zero (or end of sample if it never does).
    """
    recessions = {
        '1960-61':                ('1960-04-01', '1961-02-01'),
        '1969-70':                ('1969-12-01', '1970-11-01'),
        '1973-75 (Oil Crisis)':   ('1973-11-01', '1975-03-01'),
        '1980 (Volcker)':         ('1980-01-01', '1980-07-01'),
        '1981-82 (Volcker II)':   ('1981-07-01', '1982-11-01'),
        '1990-91 (Gulf War)':     ('1990-07-01', '1991-03-01'),
        '2001 (Dot-com)':         ('2001-03-01', '2001-11-01'),
        '2007-09 (Financial Crisis)': ('2007-12-01', '2009-06-01'),
        '2020 (COVID)':           ('2020-02-01', '2020-04-01'),
    }

    results = []
    for name, (peak_str, trough_str) in recessions.items():
        peak_date = pd.Timestamp(peak_str)
        trough_date = pd.Timestamp(trough_str)

        # Search window: NBER peak through 2 quarters after NBER trough
        window_end = trough_date + pd.DateOffset(months=6)
        mask = (df.index >= peak_date) & (df.index <= window_end)
        if mask.sum() == 0:
            continue

        # Trough = quarter with minimum CBO gap in the window
        trough_idx = df.loc[mask, 'Gap_CBO'].idxmin()

        # All gaps at the same trough quarter
        gap_cbo = df.loc[trough_idx, 'Gap_CBO']
        gap_l2 = df.loc[trough_idx, 'Gap_L2']
        gap_l1 = df.loc[trough_idx, 'Gap_L1']

        # Pre-recession peak gap: quarter before the NBER peak
        pre_peak_candidates = df.index[df.index < peak_date]
        if len(pre_peak_candidates) > 0:
            pre_peak_date = pre_peak_candidates[-1]
            peak_cbo = df.loc[pre_peak_date, 'Gap_CBO']
            peak_l1 = df.loc[pre_peak_date, 'Gap_L1']
            peak_l2 = df.loc[pre_peak_date, 'Gap_L2']
        else:
            peak_cbo = peak_l1 = peak_l2 = np.nan

        # Length: quarters from NBER peak to CBO trough
        peak_pos = df.index.get_indexer([peak_date], method='nearest')[0]
        trough_pos = df.index.get_loc(trough_idx)
        length = trough_pos - peak_pos

        # Recovery: quarters from trough until CBO gap >= 0
        post_trough = df.loc[df.index > trough_idx, 'Gap_CBO']
        recovery_quarters = post_trough[post_trough >= 0]
        if len(recovery_quarters) > 0:
            recovery_date = recovery_quarters.index[0]
            recovery = df.index.get_loc(recovery_date) - trough_pos
        else:
            recovery = len(post_trough)  # hasn't recovered by end of sample

        results.append({
            'Recession': name,
            'Trough_Date': _quarter_label(trough_idx),
            'Length': length,
            'Peak_CBO': peak_cbo,
            'Peak_L1': peak_l1,
            'Peak_L2': peak_l2,
            'Gap_CBO': gap_cbo,
            'Gap_L1': gap_l1,
            'Gap_L2': gap_l2,
            'Recovery': recovery,
        })

    return pd.DataFrame(results)


def plot_output_gaps(df, save_path=None):
    """
    Plot output gaps over time with recession shading.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel A: All three gaps
    ax1 = axes[0]
    ax1.plot(df.index, df['Gap_CBO'], 'b-', label='CBO (Production Function)', linewidth=1.5, alpha=0.8)
    ax1.plot(df.index, df['Gap_L2'], 'g--', label='L2-HP (Standard)', linewidth=1.5, alpha=0.8)
    ax1.plot(df.index, df['Gap_L1'], 'r-', label='L1-HP (Robust)', linewidth=2)
    ax1.axhline(0, color='black', linestyle=':', linewidth=0.5)

    # Shade recessions
    add_recession_shading(ax1, df)

    ax1.set_ylabel('Output Gap (%)')
    ax1.set_title('Output Gap Estimates: CBO vs. Statistical Filters')
    ax1.legend(loc='lower left')
    ax1.set_ylim(-12, 6)
    ax1.grid(True, alpha=0.3)

    # Panel B: Difference (L1 - CBO)
    ax2 = axes[1]
    diff_l1_cbo = df['Gap_L1'] - df['Gap_CBO']
    diff_l2_cbo = df['Gap_L2'] - df['Gap_CBO']

    ax2.fill_between(df.index, 0, diff_l1_cbo, where=diff_l1_cbo < 0,
                     color='red', alpha=0.3, label='L1 gap more negative than CBO')
    ax2.fill_between(df.index, 0, diff_l1_cbo, where=diff_l1_cbo >= 0,
                     color='blue', alpha=0.3, label='L1 gap less negative than CBO')
    ax2.plot(df.index, diff_l1_cbo, 'r-', linewidth=1.5, label='L1 - CBO')
    ax2.plot(df.index, diff_l2_cbo, 'g--', linewidth=1, alpha=0.7, label='L2 - CBO')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

    add_recession_shading(ax2, df)

    ax2.set_ylabel('Gap Difference (pp)')
    ax2.set_xlabel('Date')
    ax2.set_title('Difference from CBO Estimate (Filter - CBO)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


def plot_potential_output(df, save_path=None):
    """
    Plot potential output estimates (in levels) to show divergence.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Convert log trends back to levels for comparison with CBO
    # CBO is in billions, our trends are in log*100
    gdp_level = df['GDP']
    cbo_potential = df['GDP_Potential_CBO']

    # Convert filter trends: exp(trend/100) * scale_factor
    # We need to match the scale. Use the ratio at a reference point.
    ref_idx = len(df) // 2  # middle of sample
    scale_l2 = gdp_level.iloc[ref_idx] / np.exp(df['Trend_L2'].iloc[ref_idx] / 100)
    scale_l1 = gdp_level.iloc[ref_idx] / np.exp(df['Trend_L1'].iloc[ref_idx] / 100)

    potential_l2 = np.exp(df['Trend_L2'] / 100) * scale_l2
    potential_l1 = np.exp(df['Trend_L1'] / 100) * scale_l1

    ax.plot(df.index, gdp_level, 'k-', label='Actual GDP', linewidth=1, alpha=0.5)
    ax.plot(df.index, cbo_potential, 'b-', label='CBO Potential', linewidth=1.5)
    ax.plot(df.index, potential_l2, 'g--', label='L2-HP Trend', linewidth=1.5)
    ax.plot(df.index, potential_l1, 'r-', label='L1-HP Trend', linewidth=2)

    add_recession_shading(ax, df)

    ax.set_ylabel('Billions of 2017 Dollars')
    ax.set_xlabel('Date')
    ax.set_title('Potential Output Estimates: CBO vs. Statistical Filters')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


def plot_recession_zoom(df, recession_name, start, end, save_path=None):
    """
    Zoom in on a specific recession to show gap divergence.
    """
    # Extend window for context
    window_start = pd.to_datetime(start) - pd.DateOffset(years=2)
    window_end = pd.to_datetime(end) + pd.DateOffset(years=3)

    mask = (df.index >= window_start) & (df.index <= window_end)
    df_zoom = df.loc[mask]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_zoom.index, df_zoom['Gap_CBO'], 'b-', label='CBO', linewidth=2, marker='o', markersize=3)
    ax.plot(df_zoom.index, df_zoom['Gap_L2'], 'g--', label='L2-HP', linewidth=2, marker='s', markersize=3)
    ax.plot(df_zoom.index, df_zoom['Gap_L1'], 'r-', label='L1-HP', linewidth=2, marker='^', markersize=3)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

    # Shade the recession period
    ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), alpha=0.2, color='gray', label='NBER Recession')

    ax.set_ylabel('Output Gap (%)')
    ax.set_xlabel('Date')
    ax.set_title(f'Output Gap Comparison: {recession_name}')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close()


def add_recession_shading(ax, df):
    """
    Add NBER recession shading to a plot.
    """
    recession = df['Recession'].fillna(0)
    in_recession = False
    rec_start = None

    for date, val in recession.items():
        if val == 1 and not in_recession:
            rec_start = date
            in_recession = True
        elif val == 0 and in_recession:
            ax.axvspan(rec_start, date, alpha=0.15, color='gray')
            in_recession = False

    if in_recession:
        ax.axvspan(rec_start, df.index[-1], alpha=0.15, color='gray')


def run_cbo_comparison():
    """
    Main function to run the CBO comparison analysis.
    """
    print("\n" + "="*70)
    print("CBO OUTPUT GAP COMPARISON")
    print("="*70)

    # Setup paths
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(script_dir, '../../output/results'))
    figures_dir = os.path.abspath(os.path.join(script_dir, '../../output/figures'))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Download data
    df = download_cbo_data()

    # Compute output gaps
    print("\nComputing output gaps...")
    df = compute_output_gaps(df)

    # Summary statistics
    print("\n--- Overall Gap Statistics ---")
    gap_stats = compute_gap_statistics(df)
    print(gap_stats.to_string(index=False))

    # Recession vs expansion
    print("\n--- Recession vs Expansion ---")
    recession_stats = compute_recession_gaps(df)
    print(recession_stats.to_string(index=False))

    # Specific recessions
    print("\n--- Peak Gaps by Recession ---")
    specific_recessions = compute_specific_recession_gaps(df)
    print(specific_recessions.to_string(index=False))

    # Key finding
    print("\n--- Key Finding ---")
    gfc = specific_recessions[specific_recessions['Recession'].str.contains('Financial')]
    if len(gfc) > 0:
        gfc_row = gfc.iloc[0]
        print(f"2008 Financial Crisis:")
        print(f"  CBO trough gap:    {gfc_row['Gap_CBO']:.2f}%")
        print(f"  L2-HP trough gap:  {gfc_row['Gap_L2']:.2f}%")
        print(f"  L1-HP trough gap:  {gfc_row['Gap_L1']:.2f}%")
        print(f"  L1 minus CBO: {gfc_row['Gap_L1'] - gfc_row['Gap_CBO']:.2f}pp")

    # Save results
    df.to_csv(os.path.join(output_dir, 'cbo_comparison_gaps.csv'))
    gap_stats.to_csv(os.path.join(output_dir, 'cbo_comparison_stats.csv'), index=False)
    specific_recessions.to_csv(os.path.join(output_dir, 'cbo_comparison_recessions.csv'), index=False)
    print(f"\nResults saved to {output_dir}")

    # Generate figures
    print("\nGenerating figures...")
    plot_output_gaps(df, save_path=os.path.join(figures_dir, 'cbo_gap_comparison.png'))
    plot_potential_output(df, save_path=os.path.join(figures_dir, 'cbo_potential_comparison.png'))

    # Zoom plots for key recessions
    plot_recession_zoom(df, '2007-09 Financial Crisis', '2007-12-01', '2009-06-01',
                       save_path=os.path.join(figures_dir, 'cbo_zoom_2008.png'))
    plot_recession_zoom(df, '2020 COVID Recession', '2020-02-01', '2020-04-01',
                       save_path=os.path.join(figures_dir, 'cbo_zoom_2020.png'))

    print("\n" + "="*70)
    print("CBO COMPARISON COMPLETE")
    print("="*70)

    return df, gap_stats, specific_recessions


if __name__ == "__main__":
    run_cbo_comparison()
