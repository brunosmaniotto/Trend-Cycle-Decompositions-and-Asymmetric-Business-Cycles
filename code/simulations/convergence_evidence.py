"""
Monte Carlo Convergence Evidence for L1-HP Filter

Shows that L1-HP bias and RMSE decrease as sample size grows,
providing empirical evidence for consistency.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter
from models.dgps import StatisticalPluckingDGP, UCModelDGP, SharpCrashDGP, AsymmetricRWDGP


def run_convergence_experiment(dgp, sample_sizes=[50, 100, 200, 400, 800],
                                n_sim=200, lamb_l1=600, lamb_l2=1600):
    """
    For a given DGP, run L1-HP and L2-HP at increasing sample sizes.
    Track: bias, RMSE, trend correlation with true trend.

    Parameters
    ----------
    dgp : DGP object
        Data generating process with .generate(T, seed) method.
    sample_sizes : list of int
        Sample sizes to evaluate.
    n_sim : int
        Number of Monte Carlo simulations per sample size.
    lamb_l1 : float
        Smoothing parameter for L1-HP filter.
    lamb_l2 : float
        Smoothing parameter for L2-HP filter.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: T, filter, bias, rmse, trend_corr
    """
    records = []

    for T in sample_sizes:
        # Accumulators for each filter
        l1_bias_list = []
        l1_rmse_list = []
        l1_corr_list = []
        l2_bias_list = []
        l2_rmse_list = []
        l2_corr_list = []

        for sim_idx in tqdm(range(n_sim),
                            desc=f"{dgp.__class__.__name__} T={T}",
                            leave=False):
            # Generate data
            y, true_trend, true_cycle = dgp.generate(T, seed=sim_idx)

            # --- L2-HP ---
            try:
                trend_l2, cycle_l2 = l2_hp_filter(y, lamb=lamb_l2)
                valid = ~np.isnan(trend_l2)
                if valid.sum() > 1:
                    l2_bias_list.append(np.mean(trend_l2[valid] - true_trend[valid]))
                    l2_rmse_list.append(
                        np.sqrt(np.mean((trend_l2[valid] - true_trend[valid]) ** 2))
                    )
                    l2_corr_list.append(
                        np.corrcoef(trend_l2[valid], true_trend[valid])[0, 1]
                    )
            except Exception:
                pass

            # --- L1-HP ---
            try:
                trend_l1, cycle_l1 = l1_hp_filter(y, lamb=lamb_l1)
                valid = ~np.isnan(trend_l1)
                if valid.sum() > 1:
                    l1_bias_list.append(np.mean(trend_l1[valid] - true_trend[valid]))
                    l1_rmse_list.append(
                        np.sqrt(np.mean((trend_l1[valid] - true_trend[valid]) ** 2))
                    )
                    l1_corr_list.append(
                        np.corrcoef(trend_l1[valid], true_trend[valid])[0, 1]
                    )
            except Exception:
                pass

        # Average across simulations
        if l2_bias_list:
            records.append({
                'T': T,
                'filter': 'HP-L2',
                'bias': np.mean(l2_bias_list),
                'rmse': np.mean(l2_rmse_list),
                'trend_corr': np.mean(l2_corr_list)
            })
        if l1_bias_list:
            records.append({
                'T': T,
                'filter': 'HP-L1',
                'bias': np.mean(l1_bias_list),
                'rmse': np.mean(l1_rmse_list),
                'trend_corr': np.mean(l1_corr_list)
            })

    return pd.DataFrame(records)


def run_all_convergence(n_sim=200, output_dir=None):
    """
    Run convergence experiment for 4 key DGPs:
    - StatisticalPluckingDGP (asymmetric)
    - UCModelDGP (symmetric baseline)
    - SharpCrashDGP (extreme asymmetry)
    - AsymmetricRWDGP (moderate asymmetry)

    Generate:
    1. Figure: 2x2 grid, each panel shows RMSE vs T for L1 and L2
    2. Table: convergence rates (CSV)

    Save to output/figures/convergence_curves.png and
    output/results/convergence_results.csv

    Parameters
    ----------
    n_sim : int
        Number of Monte Carlo simulations per (DGP, T) combination.
    output_dir : str or None
        Root output directory. Defaults to ../../output relative to this file.

    Returns
    -------
    all_results : pd.DataFrame
        Combined convergence results across all DGPs.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output')

    fig_dir = os.path.join(output_dir, 'figures')
    res_dir = os.path.join(output_dir, 'results')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    sample_sizes = [50, 100, 200, 400, 800]

    # DGPs with their optimal L1 lambdas (from sensitivity analysis)
    dgp_configs = {
        'StatisticalPlucking': {
            'dgp': StatisticalPluckingDGP(),
            'lamb_l1': 600,
        },
        'UCModel': {
            'dgp': UCModelDGP(),
            'lamb_l1': 10,
        },
        'SharpCrash': {
            'dgp': SharpCrashDGP(),
            'lamb_l1': 1600,
        },
        'AsymmetricRW': {
            'dgp': AsymmetricRWDGP(),
            'lamb_l1': 400,
        },
    }

    print("=" * 60)
    print("CONVERGENCE EVIDENCE FOR L1-HP FILTER")
    print("=" * 60)

    all_frames = []

    for dgp_name, config in dgp_configs.items():
        print(f"\n--- {dgp_name} ({n_sim} simulations per sample size) ---")
        df = run_convergence_experiment(
            dgp=config['dgp'],
            sample_sizes=sample_sizes,
            n_sim=n_sim,
            lamb_l1=config['lamb_l1'],
            lamb_l2=1600
        )
        df['dgp'] = dgp_name
        all_frames.append(df)
        print(df[['T', 'filter', 'bias', 'rmse', 'trend_corr']].to_string(index=False))

    all_results = pd.concat(all_frames, ignore_index=True)

    # ---- Save CSV ----
    csv_path = os.path.join(res_dir, 'convergence_results.csv')
    all_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ---- Generate 2x2 Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    dgp_names = list(dgp_configs.keys())

    for idx, dgp_name in enumerate(dgp_names):
        ax = axes[idx // 2, idx % 2]
        subset = all_results[all_results['dgp'] == dgp_name]

        for filt, color, marker in [('HP-L2', 'steelblue', 'o'),
                                     ('HP-L1', 'firebrick', 's')]:
            filt_data = subset[subset['filter'] == filt]
            if not filt_data.empty:
                ax.plot(filt_data['T'], filt_data['rmse'],
                        color=color, marker=marker, linewidth=1.8,
                        markersize=6, label=filt)

        ax.set_xscale('log')
        ax.set_xlabel('Sample size (T)', fontsize=11)
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(dgp_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set x-ticks to the actual sample sizes
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels([str(s) for s in sample_sizes])

    fig.suptitle('Convergence Evidence: RMSE vs Sample Size',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig_path = os.path.join(fig_dir, 'convergence_curves.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")

    return all_results


if __name__ == '__main__':
    results = run_all_convergence(n_sim=200)
