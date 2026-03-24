"""
Summary Figures for Manuscript

1. Event-study recession plot
2. Prediction error decomposition (expansion vs recession)
3. L1-L2 gap difference bar chart by NBER recession
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter
from config import LAMBDA_L1_EMPIRICAL, LAMBDA_L2_STANDARD, NBER_RECESSIONS

# Base directory: two levels up from code/plotting/ -> Current/
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Recession labels for display (keyed by trough date string from NBER_RECESSIONS)
_RECESSION_LABELS = {
    '1961-02-01': '1960-61',
    '1970-11-01': '1969-70',
    '1975-03-01': '1973-75',
    '1980-07-01': '1980',
    '1982-11-01': '1981-82',
    '1991-03-01': '1990-91',
    '2001-11-01': '2001',
    '2009-06-01': '2007-09',
    '2020-04-01': '2020',
}


def _load_gdp_and_gaps(data_path):
    """
    Load GDP data and compute L1 and L2 output gaps.

    Returns
    -------
    df : pd.DataFrame
        DataFrame indexed by quarterly dates with columns 'GDPC1_log',
        'gap_l2', and 'gap_l1'.
    """
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if 'GDPC1_log' not in df.columns:
        print("Warning: 'GDPC1_log' column not found in data. Cannot compute gaps.")
        return None

    y = df['GDPC1_log'].dropna().values

    _, cycle_l2 = l2_hp_filter(y, lamb=LAMBDA_L2_STANDARD)
    _, cycle_l1 = l1_hp_filter(y, lamb=LAMBDA_L1_EMPIRICAL)

    idx = df['GDPC1_log'].dropna().index
    result = pd.DataFrame({
        'GDPC1_log': y,
        'gap_l2': cycle_l2,
        'gap_l1': cycle_l1,
    }, index=idx)

    return result


def _find_trough_index(df, trough_date_str):
    """
    Find the DataFrame index position closest to the given trough date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DatetimeIndex.
    trough_date_str : str
        Trough date string (e.g. '2009-06-01').

    Returns
    -------
    int or None
        Integer positional index of the trough quarter, or None if the date
        falls outside the data range.
    """
    trough_date = pd.Timestamp(trough_date_str)
    if trough_date < df.index[0] or trough_date > df.index[-1]:
        return None
    # Find the closest date in the index
    idx_pos = df.index.get_indexer([trough_date], method='nearest')[0]
    return idx_pos


def plot_recession_event_study(data_path=None, output_dir=None):
    """
    Event-study recession plot centered at each NBER trough.

    Centers each recession at t=0 (trough quarter) and plots the average
    L1 and L2 output gap paths in a window of [-4, +16] quarters, along
    with individual recession paths as thin gray lines.

    Parameters
    ----------
    data_path : str, optional
        Path to the processed FRED CSV. Defaults to
        ``data/processed/fred_data_processed.csv``.
    output_dir : str, optional
        Directory in which to save the figure. Defaults to
        ``output/figures``.
    """
    if data_path is None:
        data_path = os.path.join(_BASE_DIR, 'data', 'processed',
                                 'fred_data_processed.csv')
    if output_dir is None:
        output_dir = os.path.join(_BASE_DIR, 'output', 'figures')

    if not os.path.exists(data_path):
        print(f"Warning: Data file not found: {data_path}. Skipping recession event study.")
        return

    df = _load_gdp_and_gaps(data_path)
    if df is None:
        return

    pre_quarters = 4
    post_quarters = 16
    window = range(-pre_quarters, post_quarters + 1)

    l1_paths = []
    l2_paths = []

    for _, trough_str in NBER_RECESSIONS:
        trough_pos = _find_trough_index(df, trough_str)
        if trough_pos is None:
            continue
        start = trough_pos - pre_quarters
        end = trough_pos + post_quarters
        if start < 0 or end >= len(df):
            continue

        l1_path = df['gap_l1'].iloc[start:end + 1].values
        l2_path = df['gap_l2'].iloc[start:end + 1].values
        l1_paths.append(l1_path)
        l2_paths.append(l2_path)

    if len(l1_paths) == 0:
        print("Warning: No recession windows could be constructed. Skipping event study.")
        return

    l1_paths = np.array(l1_paths)
    l2_paths = np.array(l2_paths)
    avg_l1 = np.mean(l1_paths, axis=0)
    avg_l2 = np.mean(l2_paths, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Individual recession paths (thin gray)
    for i in range(l1_paths.shape[0]):
        ax.plot(list(window), l1_paths[i], color='gray', alpha=0.2, linewidth=0.7)

    # Average paths
    ax.plot(list(window), avg_l1, color='red', linewidth=2.0, label='L1-HP (average)')
    ax.plot(list(window), avg_l2, color='blue', linewidth=2.0, label='L2-HP (average)')

    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axvline(0, color='black', linestyle=':', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Quarters relative to trough (t = 0)')
    ax.set_ylabel('Output gap (log points)')
    ax.set_title('Recession Event Study: L1 vs L2 Output Gap')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'recession_event_study.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_decomposition(mc_results_path=None, output_dir=None):
    """
    Grouped bar chart comparing HP-L1 vs HP-L2 RMSE across asymmetric DGPs.

    Reads Section 2 Monte Carlo results and plots RMSE for the subset of
    DGPs with built-in asymmetry.

    Parameters
    ----------
    mc_results_path : str, optional
        Path to ``section2_sim_results.csv``. Defaults to
        ``output/results/section2_sim_results.csv``.
    output_dir : str, optional
        Directory in which to save the figure. Defaults to
        ``output/figures``.
    """
    if mc_results_path is None:
        mc_results_path = os.path.join(_BASE_DIR, 'output', 'results',
                                       'section2_sim_results.csv')
    if output_dir is None:
        output_dir = os.path.join(_BASE_DIR, 'output', 'figures')

    if not os.path.exists(mc_results_path):
        print(f"Warning: Monte Carlo results not found: {mc_results_path}. "
              "Skipping error decomposition plot.")
        return

    df = pd.read_csv(mc_results_path)

    # Validate expected columns
    required_cols = {'Filter', 'DGP', 'RMSE'}
    if not required_cols.issubset(df.columns):
        print(f"Warning: CSV missing required columns {required_cols - set(df.columns)}. "
              "Skipping error decomposition plot.")
        return

    # Asymmetric DGP identifiers (partial matches against the DGP column)
    asym_dgp_keys = [
        '7_StatPlucking',
        '8_SharpCrash',
        '4b_AsymmetricRW',
        '10_AsymGarch',
        '5_SkewedUC',
        '11_SkewFatTail',
    ]

    # Friendly display labels
    asym_labels = {
        '7_StatPlucking': 'Stat. Plucking',
        '8_SharpCrash': 'Sharp Crash',
        '4b_AsymmetricRW': 'Asymmetric RW',
        '10_AsymGarch': 'Asym. GARCH',
        '5_SkewedUC': 'Skewed UC',
        '11_SkewFatTail': 'Skew-Fat Tail',
    }

    hp_l2 = df[(df['Filter'] == 'HP-L2') & (df['DGP'].isin(asym_dgp_keys))].copy()
    hp_l1 = df[(df['Filter'] == 'HP-L1') & (df['DGP'].isin(asym_dgp_keys))].copy()

    if hp_l2.empty or hp_l1.empty:
        print("Warning: Could not find HP-L1 / HP-L2 rows for asymmetric DGPs. "
              "Skipping error decomposition plot.")
        return

    # Merge on DGP for side-by-side comparison
    merged = pd.merge(
        hp_l2[['DGP', 'RMSE']].rename(columns={'RMSE': 'RMSE_L2'}),
        hp_l1[['DGP', 'RMSE']].rename(columns={'RMSE': 'RMSE_L1'}),
        on='DGP',
    )

    # Preserve the order defined above
    merged['DGP'] = pd.Categorical(merged['DGP'], categories=asym_dgp_keys, ordered=True)
    merged = merged.sort_values('DGP').reset_index(drop=True)
    merged['label'] = merged['DGP'].map(asym_labels)

    x = np.arange(len(merged))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - bar_width / 2, merged['RMSE_L2'], bar_width, label='HP-L2', color='steelblue')
    ax.bar(x + bar_width / 2, merged['RMSE_L1'], bar_width, label='HP-L1', color='firebrick')

    ax.set_xticks(x)
    ax.set_xticklabels(merged['label'], rotation=25, ha='right')
    ax.set_ylabel('RMSE')
    ax.set_title('L1 vs L2 RMSE by DGP Type')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'error_decomposition.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def plot_gap_difference_bar(data_path=None, output_dir=None):
    """
    Bar chart of trough-gap difference (L1 - L2) for each NBER recession.

    For each recession, finds the quarter closest to the NBER trough date
    and computes the difference between the L1 and L2 output gaps.

    Parameters
    ----------
    data_path : str, optional
        Path to the processed FRED CSV. Defaults to
        ``data/processed/fred_data_processed.csv``.
    output_dir : str, optional
        Directory in which to save the figure. Defaults to
        ``output/figures``.
    """
    if data_path is None:
        data_path = os.path.join(_BASE_DIR, 'data', 'processed',
                                 'fred_data_processed.csv')
    if output_dir is None:
        output_dir = os.path.join(_BASE_DIR, 'output', 'figures')

    if not os.path.exists(data_path):
        print(f"Warning: Data file not found: {data_path}. Skipping gap difference bar chart.")
        return

    df = _load_gdp_and_gaps(data_path)
    if df is None:
        return

    labels = []
    differences = []

    for _, trough_str in NBER_RECESSIONS:
        trough_pos = _find_trough_index(df, trough_str)
        if trough_pos is None:
            continue

        l1_gap = df['gap_l1'].iloc[trough_pos]
        l2_gap = df['gap_l2'].iloc[trough_pos]
        diff = l1_gap - l2_gap

        label = _RECESSION_LABELS.get(trough_str, trough_str[:4])
        labels.append(label)
        differences.append(diff)

    if len(labels) == 0:
        print("Warning: No recession troughs found in data range. "
              "Skipping gap difference bar chart.")
        return

    differences = np.array(differences)
    colors = ['firebrick' if d < 0 else 'steelblue' for d in differences]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), differences, color=colors, edgecolor='black',
           linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Gap difference (L1 - L2, log points)')
    ax.set_title('Trough Output Gap Difference by NBER Recession')
    ax.grid(axis='y', alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'gap_difference_bar.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


def generate_summary_figures(data_path=None, output_dir=None):
    """
    Generate all three summary figures.

    Parameters
    ----------
    data_path : str, optional
        Path to the processed FRED CSV. Defaults to
        ``data/processed/fred_data_processed.csv``.
    output_dir : str, optional
        Directory in which to save the figures. Defaults to
        ``output/figures``.
    """
    if data_path is None:
        data_path = os.path.join(_BASE_DIR, 'data', 'processed',
                                 'fred_data_processed.csv')
    if output_dir is None:
        output_dir = os.path.join(_BASE_DIR, 'output', 'figures')

    print("=" * 60)
    print("Generating summary figures")
    print("=" * 60)

    print("\n[1/3] Recession event study...")
    plot_recession_event_study(data_path=data_path, output_dir=output_dir)

    print("\n[2/3] Error decomposition (MC results)...")
    plot_error_decomposition(output_dir=output_dir)

    print("\n[3/3] Gap difference bar chart...")
    plot_gap_difference_bar(data_path=data_path, output_dir=output_dir)

    print("\nSummary figures complete.")


if __name__ == '__main__':
    generate_summary_figures()
