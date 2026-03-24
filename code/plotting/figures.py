import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Base directory (two levels up from this file: code/plotting/ -> Current/)
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def set_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('husl')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11


def plot_gdp_cycles(results_path=None, figures_dir=None):
    """
    Plot GDP cycles from empirical analysis.
    """
    if results_path is None:
        results_path = os.path.join(_BASE_DIR, 'output', 'results', 'gdp_filter_results.csv')
    if figures_dir is None:
        figures_dir = os.path.join(_BASE_DIR, 'output', 'figures')

    if not os.path.exists(results_path):
        print(f"GDP results not found for plotting: {results_path}")
        return

    df = pd.read_csv(results_path, index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Cycle_HP_L2'], label='L2-HP (Standard)', alpha=0.7)
    ax.plot(df.index, df['Cycle_HP_L1'], label='L1-HP (Robust)', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)

    ax.set_title('US Real GDP Cycle: L1 vs L2 Filter')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'gdp_cycles.png'), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(figures_dir, 'gdp_cycles.png')}")


def plot_sp500_cycles(extended_results, figures_dir=None):
    """
    Plot S&P 500 cycles from extended empirical analysis.
    """
    if 'SP500_Cycle_L2' not in extended_results or 'SP500_Cycle_L1' not in extended_results:
        print("S&P 500 cycle data not found in extended_results.")
        return

    if figures_dir is None:
        figures_dir = os.path.join(_BASE_DIR, 'output', 'figures')

    # Load dates from processed data
    processed_data_path = os.path.join(_BASE_DIR, 'data', 'processed', 'fred_data_processed.csv')
    if not os.path.exists(processed_data_path):
        print(f"Processed data not found: {processed_data_path}")
        return

    df_dates = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

    # Align cycles with dates (SP500 has shorter history)
    if 'SP500_log' in df_dates.columns:
        sp500_dates = df_dates['SP500_log'].dropna().index
    else:
        # Fallback if column missing (unlikely given extended_results has data)
        sp500_dates = df_dates.index[-len(extended_results['SP500_Cycle_L2']):]

    sp500_cycle_l2 = extended_results['SP500_Cycle_L2']
    sp500_cycle_l1 = extended_results['SP500_Cycle_L1']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sp500_dates, sp500_cycle_l2, label='L2-HP (Standard)', alpha=0.7)
    ax.plot(sp500_dates, sp500_cycle_l1, label='L1-HP (Robust)', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)

    ax.set_title('US S&P 500 Real Cycle: L1 vs L2 Filter')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, 'sp500_cycles.png'), dpi=300)
    plt.close()
    print(f"Saved: {os.path.join(figures_dir, 'sp500_cycles.png')}")