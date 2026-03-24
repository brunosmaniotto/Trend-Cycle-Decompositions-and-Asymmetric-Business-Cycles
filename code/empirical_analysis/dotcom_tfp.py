"""
Dot-Com Bubble and TFP Break Analysis (1995-2005)

Focused analysis asking: does L1-HP's piecewise-linear trend
naturally detect the TFP growth slowdown around 2000-2001?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter
from config import LAMBDA_L1_EMPIRICAL, LAMBDA_L2_STANDARD


def run_dotcom_analysis(data_path='../../data/processed/fred_data_processed.csv'):
    """
    Analyze L1 vs L2 trend behavior during the dot-com period.

    Generates a 3-panel figure:
    1. GDP trend zoomed to 1995-2005: L1 vs L2 trend over observed GDP
    2. Productivity trend (OPHNFB_log) showing break detection
    3. Output gap comparison during the bubble period

    Key question: does L1's piecewise-linear trend naturally detect
    the TFP growth slowdown that occurred around 2000-2001?
    """
    # Load data
    data_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
    if not os.path.exists(data_full_path):
        print(f"Data file not found: {data_full_path}")
        return None

    df = pd.read_csv(data_full_path, index_col=0, parse_dates=True)

    script_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'output'))
    fig_dir = os.path.join(output_dir, 'figures')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Apply filters to FULL sample (filter needs full context)
    gdp_log = df['GDPC1_log'].values
    dates = df.index

    trend_l2, cycle_l2 = l2_hp_filter(gdp_log, lamb=LAMBDA_L2_STANDARD)
    trend_l1, cycle_l1 = l1_hp_filter(gdp_log, lamb=LAMBDA_L1_EMPIRICAL)

    gap_l2 = cycle_l2 * 100
    gap_l1 = cycle_l1 * 100

    # Zoom mask for 1993-2007 (show broader context around 1995-2005)
    zoom_mask = (dates >= '1993-01-01') & (dates <= '2007-01-01')

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # Panel 1: GDP Trend
    ax1 = axes[0]
    ax1.plot(dates[zoom_mask], gdp_log[zoom_mask], 'k-', alpha=0.4, label='Log Real GDP', linewidth=1)
    ax1.plot(dates[zoom_mask], trend_l2[zoom_mask], 'b-', label='L2-HP Trend', linewidth=2)
    ax1.plot(dates[zoom_mask], trend_l1[zoom_mask], 'r-', label='L1-HP Trend', linewidth=2)
    # Mark recession
    ax1.axvspan(pd.Timestamp('2001-03-01'), pd.Timestamp('2001-11-01'), alpha=0.2, color='gray', label='NBER Recession')
    ax1.set_title('GDP Trend: Dot-Com Period (1993-2007)', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Compute trend growth rates (annualized)
    # Trend growth = diff(trend) * 400 (quarterly to annual %)
    trend_growth_l2 = np.diff(trend_l2) * 400
    trend_growth_l1 = np.diff(trend_l1) * 400

    # Panel 2: Productivity trend (if available)
    ax2 = axes[1]
    if 'OPHNFB_log' in df.columns:
        prod = df['OPHNFB_log'].dropna()
        prod_dates = prod.index
        prod_vals = prod.values

        prod_trend_l2, _ = l2_hp_filter(prod_vals, lamb=LAMBDA_L2_STANDARD)
        prod_trend_l1, _ = l1_hp_filter(prod_vals, lamb=LAMBDA_L1_EMPIRICAL)

        prod_zoom = (prod_dates >= '1993-01-01') & (prod_dates <= '2007-01-01')

        ax2.plot(prod_dates[prod_zoom], prod_vals[prod_zoom], 'k-', alpha=0.4, label='Log Productivity', linewidth=1)
        ax2.plot(prod_dates[prod_zoom], prod_trend_l2[prod_zoom], 'b-', label='L2-HP Trend', linewidth=2)
        ax2.plot(prod_dates[prod_zoom], prod_trend_l1[prod_zoom], 'r-', label='L1-HP Trend', linewidth=2)
        ax2.axvspan(pd.Timestamp('2001-03-01'), pd.Timestamp('2001-11-01'), alpha=0.2, color='gray')
        ax2.set_title('Labor Productivity Trend: TFP Break Detection', fontsize=13)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Productivity data not available', transform=ax2.transAxes, ha='center')

    # Panel 3: Output gap comparison
    ax3 = axes[2]
    ax3.plot(dates[zoom_mask], gap_l2[zoom_mask], 'b-', label='L2-HP Gap (%)', linewidth=1.5)
    ax3.plot(dates[zoom_mask], gap_l1[zoom_mask], 'r-', label='L1-HP Gap (%)', linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.axvspan(pd.Timestamp('2001-03-01'), pd.Timestamp('2001-11-01'), alpha=0.2, color='gray', label='NBER Recession')
    ax3.set_title('Output Gap Comparison: Dot-Com Period', fontsize=13)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Output Gap (%)')

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'dotcom_tfp_analysis.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # Save summary statistics
    # Find the pre-break and post-break trend growth rates
    break_date = pd.Timestamp('2001-01-01')
    pre_mask = (dates[1:] >= '1995-01-01') & (dates[1:] < break_date)
    post_mask = (dates[1:] >= break_date) & (dates[1:] <= '2005-01-01')

    summary = {
        'pre_break_growth_L2': trend_growth_l2[pre_mask].mean() if pre_mask.any() else np.nan,
        'pre_break_growth_L1': trend_growth_l1[pre_mask].mean() if pre_mask.any() else np.nan,
        'post_break_growth_L2': trend_growth_l2[post_mask].mean() if post_mask.any() else np.nan,
        'post_break_growth_L1': trend_growth_l1[post_mask].mean() if post_mask.any() else np.nan,
    }
    summary['growth_change_L2'] = summary['post_break_growth_L2'] - summary['pre_break_growth_L2']
    summary['growth_change_L1'] = summary['post_break_growth_L1'] - summary['pre_break_growth_L1']

    summary_df = pd.DataFrame([summary])
    csv_path = os.path.join(results_dir, 'dotcom_tfp_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print(f"\nDot-Com TFP Analysis Summary:")
    print(f"  Pre-break trend growth (ann.%): L2={summary['pre_break_growth_L2']:.2f}, L1={summary['pre_break_growth_L1']:.2f}")
    print(f"  Post-break trend growth (ann.%): L2={summary['post_break_growth_L2']:.2f}, L1={summary['post_break_growth_L1']:.2f}")
    print(f"  Growth change: L2={summary['growth_change_L2']:.2f}pp, L1={summary['growth_change_L1']:.2f}pp")

    return summary
