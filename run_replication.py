"""
Master Replication Script for "Trend-Cycle Decompositions and Asymmetric Business Cycles"

This script executes the complete analysis pipeline:
1. Data download and preprocessing (FRED)
2. Section 4: Statistical Monte Carlo simulations (12 DGPs)
3. Section 5: Structural model simulations (9 models)
4. Section 6: Empirical analysis (GDP, Policy, CBO, Extended, Real-time)
5. Appendix: Mollified filters, Bootstrap coverage, ABK demos
6. Figure generation

Usage:
    python run_replication.py           # Full replication (slow, ~30-60 min)
    python run_replication.py --quick   # Quick check (~5 min)
"""

import sys
import os
import argparse

# Add code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from config import LAMBDA_L1_EMPIRICAL, LAMBDA_L2_STANDARD
from data_processing.downloader import download_fred_data
from data_processing.preprocessor import preprocess_fred_data
from simulations.monte_carlo import run_monte_carlo_experiment
from simulations.structural_experiments import run_structural_experiments
from simulations.convergence_evidence import run_all_convergence
from simulations.sensitivity import run_lambda_selection_experiments, run_l2_lambda_selection_experiments
from models.dgps import (
    StatisticalPluckingDGP, RandomWalkDGP, UCModelDGP,
    FatTailedRWDGP, SkewedUCDGP, DriftRandomWalkDGP, AsymmetricRWDGP,
    PhaseShiftDGP, SharpCrashDGP, GarchRWDGP, AsymmetricGarchDGP, SkewedFatTailDGP
)
from filters.hp import l1_hp_filter, l2_hp_filter
from filters.hamilton import l1_hamilton_filter, l2_hamilton_filter
from empirical_analysis.gdp_analysis import run_gdp_analysis
from empirical_analysis.policy import run_policy_analysis, compute_taylor_rule_counterfactuals
from empirical_analysis.extended_analysis import run_extended_analysis
from empirical_analysis.inflation_forecasting import run_inflation_forecasting
from empirical_analysis.dotcom_tfp import run_dotcom_analysis
from empirical_analysis.cbo_comparison import run_cbo_comparison
from empirical_analysis.international_evidence import run_international_experiments
from simulations.endpoint_bias import run_endpoint_experiments
from simulations.hamilton_experiments import run_hamilton_experiments
from simulations.robustness_checks import run_robustness_experiments
from simulations.real_time_analysis import run_realtime_analysis
from simulations.run_bootstrap_demo import run_bootstrap_demo
from utils.specification_test import recommend_filter
from parameters.calibration import run_calibration_analysis
from plotting.figures import set_style, plot_gdp_cycles, plot_sp500_cycles
from plotting.summary_figures import generate_summary_figures
from plotting.section_figures import (
    generate_section5_figure, generate_section6_gdp_figure,
    generate_section6_taylor_figure
)
from empirical_analysis.hamilton_empirical import run_hamilton_empirical
from simulations.lp_experiment import run_lp_joint_experiment


def run_section2_simulations(n_sim=500, T=200):
    """
    Run full Section 2 Monte Carlo simulations across all 12 DGPs.
    """
    import pandas as pd
    from tqdm import tqdm

    print("\n" + "="*60)
    print("SECTION 2: Statistical Benchmark Simulations")
    print("="*60)

    # Define all DGPs
    dgps = {
        '1_RandomWalk': RandomWalkDGP(),
        '2_DriftRW': DriftRandomWalkDGP(),
        '3_UCModel': UCModelDGP(),
        '4_FatTailedRW': FatTailedRWDGP(),
        '4b_AsymmetricRW': AsymmetricRWDGP(),
        '5_SkewedUC': SkewedUCDGP(),
        '6_PhaseShift': PhaseShiftDGP(),
        '7_StatPlucking': StatisticalPluckingDGP(),
        '8_SharpCrash': SharpCrashDGP(),
        '9_GarchRW': GarchRWDGP(),
        '10_AsymGarch': AsymmetricGarchDGP(),
        '11_SkewFatTail': SkewedFatTailDGP()
    }

    # L1 lambda for Table 1: fixed at 600 to match the paper's description
    L1_LAMBDA = 600

    all_summaries = []

    for name, dgp in dgps.items():
        print(f"\n--- Running {name} ({n_sim} simulations) ---")

        filters_for_run = {
            'HP-L2': lambda y, lamb=1600: l2_hp_filter(y, lamb=1600),
            'HP-L1': lambda y, lamb=L1_LAMBDA: l1_hp_filter(y, lamb=L1_LAMBDA),
            'Hamilton-L2': l2_hamilton_filter,
            'Hamilton-L1': l1_hamilton_filter
        }

        summary, _ = run_monte_carlo_experiment(dgp, filters_for_run, T=T, n_sim=n_sim, lamb=1600)
        summary['DGP'] = name
        all_summaries.append(summary)
        print(summary[['Filter', 'Bias', 'RMSE', 'Cycle Skewness']])

    # Save results
    full_results = pd.concat(all_summaries, ignore_index=True)
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'results')
    os.makedirs(output_dir, exist_ok=True)
    full_results.to_csv(os.path.join(output_dir, 'section2_sim_results.csv'), index=False)
    print(f"\nSection 2 results saved to {output_dir}/section2_sim_results.csv")

    return full_results


def run_mollified_simulations(n_sim=100, T=100):
    """
    Run mollified filter simulations (Huber, Elastic-Net HP variants).
    """
    import pandas as pd
    from filters.robust_smooth import huber_hp_filter, elastic_hp_filter

    print("\n" + "="*60)
    print("APPENDIX: Mollified Filter Simulations")
    print("="*60)

    dgps = {
        'Plucking': StatisticalPluckingDGP(),
        'FatTailed': FatTailedRWDGP()
    }

    results = []

    for name, dgp in dgps.items():
        print(f"\n--- Testing {name} ---")

        filters = {
            'HP-L2': lambda y, **kwargs: l2_hp_filter(y, lamb=1600),
            'HP-L1': lambda y, **kwargs: l1_hp_filter(y, lamb=600),
            'Huber-HP (d=1)': lambda y, **kwargs: huber_hp_filter(y, lamb=1000, delta=1.0),
            'Huber-HP (d=0.1)': lambda y, **kwargs: huber_hp_filter(y, lamb=1000, delta=0.1),
            'Elastic-HP (a=0.5)': lambda y, **kwargs: elastic_hp_filter(y, lamb=1000, alpha=0.5)
        }

        summary, _ = run_monte_carlo_experiment(dgp, filters, T=T, n_sim=n_sim, lamb=1600)
        summary['DGP'] = name
        results.append(summary)
        print(summary[['Filter', 'Bias', 'RMSE', 'Cycle Skewness']])

    final_df = pd.concat(results, ignore_index=True)
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'results')
    os.makedirs(output_dir, exist_ok=True)
    final_df.to_csv(os.path.join(output_dir, 'mollified_sim_results.csv'), index=False)
    print(f"\nMollified results saved to {output_dir}/mollified_sim_results.csv")

    return final_df


def generate_all_figures():
    """
    Generate all figures for the paper.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal

    from models.dgps import (
        RandomWalkDGP, DriftRandomWalkDGP, UCModelDGP,
        FatTailedRWDGP, AsymmetricRWDGP, SkewedUCDGP,
        PhaseShiftDGP, StatisticalPluckingDGP, SharpCrashDGP,
        GarchRWDGP, AsymmetricGarchDGP, SkewedFatTailDGP
    )
    from models.structural import (
        PluckingModel, NetworkModel, HANKModel, GranularStructuralModel,
        FinancialAcceleratorModel, ZLBModel, AsymmetricRBCModel,
        UncertaintyShocksModel, HysteresisModel
    )
    from filters.asymmetric_bk import asymmetric_bk_filter, standard_bk_filter
    from utils.bootstrap import bootstrap_l1_hp

    print("\n" + "="*60)
    print("GENERATING ALL FIGURES")
    print("="*60)

    base_dir = os.path.dirname(__file__)
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 1. DGP Simulation Plots ---
    print("\n[Figures] DGP Simulations...")
    sim_dir = os.path.join(base_dir, 'output', 'figures', 'simulations')
    os.makedirs(sim_dir, exist_ok=True)

    dgps = {
        'RandomWalk': RandomWalkDGP(),
        'DriftRandomWalk': DriftRandomWalkDGP(),
        'UCModel': UCModelDGP(),
        'FatTailedRW': FatTailedRWDGP(),
        'AsymmetricRW': AsymmetricRWDGP(),
        'SkewedUC': SkewedUCDGP(),
        'PhaseShift': PhaseShiftDGP(),
        'Plucking': StatisticalPluckingDGP(),
        'SharpCrash': SharpCrashDGP(),
        'GARCH_RW': GarchRWDGP(),
        'AsymmetricGARCH': AsymmetricGarchDGP(),
        'SkewedFatTail': SkewedFatTailDGP()
    }

    for name, dgp in dgps.items():
        np.random.seed(42)
        try:
            y, true_trend, true_cycle = dgp.generate(T=200)
            l2_trend, _ = l2_hp_filter(y, lamb=1600)
            l1_trend, _ = l1_hp_filter(y, lamb=600)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y, color='silver', alpha=0.5, label='Observed Data', linewidth=1)
            ax.plot(true_trend, color='black', linestyle='--', label='True Trend', linewidth=2)
            ax.plot(l2_trend, color='tab:blue', label='L2-HP (Standard)', linewidth=1.5)
            ax.plot(l1_trend, color='tab:red', label='L1-HP (Robust)', linewidth=1.5)
            ax.set_title(f'Simulation: {name}', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(sim_dir, f'sim_{name}.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"  Error plotting {name}: {e}")

    # --- 2. Structural Model Plots ---
    print("[Figures] Structural Models...")
    struct_dir = os.path.join(base_dir, 'output', 'figures', 'structural')
    os.makedirs(struct_dir, exist_ok=True)

    models = {
        'Plucking': PluckingModel(),
        'Network': NetworkModel(),
        'HANK': HANKModel(),
        'Granular': GranularStructuralModel(),
        'FinancialAccelerator': FinancialAcceleratorModel(),
        'ZLB': ZLBModel(),
        'AsymmetricRBC': AsymmetricRBCModel(),
        'UncertaintyShocks': UncertaintyShocksModel(),
        'Hysteresis': HysteresisModel(),
    }

    for name, model in models.items():
        np.random.seed(42)
        try:
            sim_data = model.simulate(T=200)
            y = sim_data['output']
            true_trend = sim_data['potential']

            l2_trend, _ = l2_hp_filter(y, lamb=1600)
            l1_trend, _ = l1_hp_filter(y, lamb=600)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y, color='silver', alpha=0.5, label='Observed Output', linewidth=1)
            ax.plot(true_trend, color='black', linestyle='--', label='True Potential', linewidth=2)
            ax.plot(l2_trend, color='tab:blue', label='L2-HP Estimate', linewidth=1.5)
            ax.plot(l1_trend, color='tab:red', label='L1-HP Estimate', linewidth=1.5)
            ax.set_title(f'Structural Model: {name}', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(struct_dir, f'struct_{name}.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"  Error plotting {name}: {e}")

    # --- 3. ABK Demo Plot ---
    print("[Figures] ABK Demo...")
    fig_dir = os.path.join(base_dir, 'output', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    dgp = StatisticalPluckingDGP()
    y, true_trend, true_cycle = dgp.generate(200, seed=42)
    bk_cycle = standard_bk_filter(y)
    abk_trend, abk_cycle = asymmetric_bk_filter(y, method='L1')

    plt.figure(figsize=(12, 6))
    plt.plot(true_cycle, 'k--', label='True Plucking Cycle', alpha=0.5)
    plt.plot(bk_cycle, 'b-', label='Standard BK (Symmetric)', alpha=0.7)
    plt.plot(abk_cycle, 'r-', label='Asymmetric BK (Split Basis)', alpha=0.7)
    plt.axhline(0, color='k', linestyle=':', alpha=0.3)
    plt.title('Asymmetric Band-Pass Filtering: Plucking Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, 'abk_demo.png'), dpi=300)
    plt.close()

    # --- 4. ABK Appendix Plots ---
    print("[Figures] ABK Appendix...")
    appendix_dir = os.path.join(base_dir, 'output', 'figures', 'appendix')
    os.makedirs(appendix_dir, exist_ok=True)

    # Sawtooth cycle demonstration
    np.random.seed(42)
    T = 200
    t = np.arange(T)
    raw_cycle = signal.sawtooth(2 * np.pi * t / 40, width=0.8)
    true_cycle_saw = raw_cycle * 2
    trend_saw = 0.5 * t
    y_saw = trend_saw + true_cycle_saw + np.random.randn(T) * 0.5

    import pandas as pd
    bk_cycle_saw = standard_bk_filter(pd.Series(y_saw), low=6, high=32, K=12)
    _, abk_cycle_saw = asymmetric_bk_filter(y_saw, low=6, high=32, n_freqs=30, method='L1')

    start, end = 15, 185
    plt.figure(figsize=(10, 6))
    plt.plot(t[start:end], true_cycle_saw[start:end], 'k--', label='True Asymmetric Cycle', alpha=0.6)
    plt.plot(t[start:end], bk_cycle_saw[start:end], label='Standard BK (Symmetric)', linewidth=1.5)
    plt.plot(t[start:end], abk_cycle_saw[start:end], label='Asymmetric BK (Split Basis)', linewidth=1.5, color='tab:red')
    plt.title('Filter Performance on Asymmetric (Sawtooth) Cycles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(appendix_dir, 'abk_time_domain.png'), dpi=300)
    plt.close()

    # Basis visualization
    t_short = np.arange(100)
    omega = 2 * np.pi / 40
    sin_wave = np.sin(omega * t_short)
    sin_pos = np.maximum(sin_wave, 0)
    sin_neg = np.maximum(-sin_wave, 0)

    plt.figure(figsize=(10, 4))
    plt.plot(t_short, sin_wave, 'k:', label='Original Sine', alpha=0.3)
    plt.plot(t_short, sin_pos, label='Positive Basis (sin+)', color='tab:red')
    plt.plot(t_short, -sin_neg, label='Negative Basis (sin-)', color='tab:blue', linestyle='--')
    plt.title('Split-Basis Decomposition: Allowing Asymmetric Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(appendix_dir, 'abk_basis.png'), dpi=300)
    plt.close()

    # --- 5. Bootstrap Uncertainty Plot ---
    print("[Figures] Bootstrap Uncertainty...")

    def plot_bootstrap(ax, time, y, trend, lower, upper, title):
        ax.plot(time, y, 'k.', alpha=0.2, label='Data', markersize=2)
        ax.plot(time, trend, 'r-', linewidth=1.5, label='L1 Trend')
        ax.fill_between(time, lower, upper, color='red', alpha=0.2, label='90% CI')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    # Case 1: Statistical Plucking
    dgp = StatisticalPluckingDGP()
    y_pluck, _, _ = dgp.generate(100, seed=42)
    trend_p, low_p, high_p, _ = bootstrap_l1_hp(y_pluck, lamb=600, n_boot=50)
    plot_bootstrap(axes[0], np.arange(100), y_pluck, trend_p, low_p, high_p,
                   "Statistical Plucking DGP: L1 Trend Uncertainty")

    # Case 2: Financial Accelerator
    model = FinancialAcceleratorModel()
    sim_data = model.simulate(T=100)
    y_fin = sim_data['output']
    trend_f, low_f, high_f, _ = bootstrap_l1_hp(y_fin, lamb=600, n_boot=50)
    plot_bootstrap(axes[1], np.arange(100), y_fin, trend_f, low_f, high_f,
                   "Financial Accelerator Model: L1 Trend Uncertainty")

    # Case 3: Real GDP
    data_path = os.path.join(base_dir, 'data', 'processed', 'fred_data_processed.csv')
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        if 'GDPC1_log' in df.columns:
            y_gdp_recent = df['GDPC1_log'].values[-100:]
            dates_recent = df.index[-100:]
            trend_g, low_g, high_g, _ = bootstrap_l1_hp(y_gdp_recent, lamb=600, n_boot=50)
            axes[2].plot(dates_recent, y_gdp_recent, 'k.', alpha=0.2, label='Data')
            axes[2].plot(dates_recent, trend_g, 'r-', linewidth=1.5, label='L1 Trend')
            axes[2].fill_between(dates_recent, low_g, high_g, color='red', alpha=0.2, label='90% CI')
            axes[2].set_title("US Real GDP (Recent): L1 Trend Uncertainty")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'bootstrap_uncertainty.png'), dpi=300)
    plt.close()

    print("[Figures] All figures generated successfully!")


def main():
    parser = argparse.ArgumentParser(description='Replication script for Trend-Cycle Decompositions paper')
    parser.add_argument('--quick', action='store_true', help='Run quick check (reduced simulations)')
    args = parser.parse_args()

    print("="*60)
    print("TREND-CYCLE DECOMPOSITIONS AND ASYMMETRIC BUSINESS CYCLES")
    print("Full Replication Pipeline")
    print("="*60)

    if args.quick:
        print("\n*** QUICK MODE: Running abbreviated simulations ***\n")
        n_sim_sec2, n_reps_sec3, n_sim_moll = 50, 5, 25
    else:
        print("\n*** FULL MODE: Running complete simulations ***\n")
        n_sim_sec2, n_reps_sec3, n_sim_moll = 500, 50, 100

    set_style()

    # Define paths
    base_dir = os.path.dirname(__file__)
    processed_data_path = os.path.join(base_dir, 'data', 'processed', 'fred_data_processed.csv')

    # =========================================================================
    # STEP 1: Data Processing
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: Data Processing")
    print("="*60)
    try:
        raw_df = download_fred_data(output_dir=os.path.join(base_dir, 'data', 'raw'))
        processed_df = preprocess_fred_data(raw_df, output_dir=os.path.join(base_dir, 'data', 'processed'))
    except Exception as e:
        print(f"Skipping data download (API error?): {e}")
        print("Proceeding with existing data if available.")

    # =========================================================================
    # STEP 2: Section 2 - Statistical Monte Carlo Simulations
    # =========================================================================
    run_section2_simulations(n_sim=n_sim_sec2, T=200)

    # =========================================================================
    # STEP 2b: Oracle Lambda Selection (Table 2)
    # =========================================================================
    print("\n" + "="*60)
    print("LAMBDA SELECTION: Oracle Optimal Lambda by DGP")
    print("="*60)
    n_sim_lambda = 50 if args.quick else 500
    try:
        run_lambda_selection_experiments(n_sim=n_sim_lambda, T=200)
        run_l2_lambda_selection_experiments(n_sim=n_sim_lambda, T=200)
    except Exception as e:
        print(f"  Lambda selection failed: {e}")

    # =========================================================================
    # STEP 2c: Lp Joint (p, lambda) Experiment (Appendix C)
    # =========================================================================
    print("\n" + "="*60)
    print("Lp EXPERIMENT: Joint (p, lambda) Grid Search")
    print("="*60)
    n_sim_lp = 10 if args.quick else 50
    try:
        run_lp_joint_experiment(base_dir, n_sim=n_sim_lp, T=200)
    except Exception as e:
        print(f"  Lp joint experiment failed: {e}")

    # =========================================================================
    # STEP 3: Section 3 - Structural Model Experiments
    # =========================================================================
    print("\n" + "="*60)
    print("SECTION 3: Structural Model Experiments")
    print("="*60)
    summary_struct = run_structural_experiments(T=200, n_reps=n_reps_sec3)

    # =========================================================================
    # STEP 4: Appendix - Mollified Filter Simulations
    # =========================================================================
    run_mollified_simulations(n_sim=n_sim_moll, T=100)

    # =========================================================================
    # STEP 5: Section 4 - Empirical Analysis
    # =========================================================================
    print("\n" + "="*60)
    print("SECTION 4: Empirical Analysis")
    print("="*60)

    print("\n[Empirical] GDP Analysis...")
    gdp_results, gdp_stats = run_gdp_analysis(data_path=processed_data_path)

    print("\n[Empirical] Policy Analysis (Taylor Rule, Phillips Curve)...")
    policy_df, pc_l2_results, pc_l1_results = run_policy_analysis(data_path=processed_data_path)

    print("\n[Empirical] Taylor Rule Counterfactuals...")
    try:
        taylor_cf = compute_taylor_rule_counterfactuals(data_path=processed_data_path)
    except Exception as e:
        print(f"  Taylor counterfactuals failed: {e}")

    print("\n[Empirical] Extended Analysis (NAIRU, TFP, Okun, etc.)...")
    extended_results = run_extended_analysis(data_path=processed_data_path)

    print("\n[Empirical] Inflation Forecasting...")
    try:
        inflation_rmsfe = run_inflation_forecasting(data_path=processed_data_path)
    except Exception as e:
        print(f"  Inflation forecasting failed: {e}")

    print("\n[Empirical] Dot-Com / TFP Break Analysis...")
    try:
        dotcom_results = run_dotcom_analysis(data_path=processed_data_path)
    except Exception as e:
        print(f"  Dot-com analysis failed: {e}")

    print("\n[Empirical] CBO Output Gap Comparison...")
    try:
        run_cbo_comparison()
    except Exception as e:
        print(f"  CBO comparison failed: {e}")

    print("\n[Empirical] Real-Time Revision Analysis (ALFRED)...")
    try:
        run_realtime_analysis()
    except Exception as e:
        print(f"  Real-time analysis failed: {e}")

    print("\n[Empirical] Bootstrap Coverage Demo...")
    try:
        run_bootstrap_demo()
    except Exception as e:
        print(f"  Bootstrap demo failed: {e}")

    # =========================================================================
    # STEP 5a: Hamilton Empirical Analysis (Appendix D)
    # =========================================================================
    print("\n[Empirical] Hamilton Filter Empirical Exercises...")
    try:
        run_hamilton_empirical(base_dir)
    except Exception as e:
        print(f"  Hamilton empirical failed: {e}")

    # =========================================================================
    # STEP 5b: Specification Test Demo
    # =========================================================================
    print("\n" + "="*60)
    print("SPECIFICATION TEST: L1 vs L2 on US GDP")
    print("="*60)
    try:
        import pandas as pd
        if os.path.exists(processed_data_path):
            df_spec = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
            if 'GDPC1_log' in df_spec.columns:
                spec_result = recommend_filter(df_spec['GDPC1_log'].values)
                print(spec_result['summary'])
    except Exception as e:
        print(f"  Specification test failed: {e}")

    # =========================================================================
    # STEP 5c: Convergence Evidence
    # =========================================================================
    print("\n" + "="*60)
    print("CONVERGENCE EVIDENCE")
    print("="*60)
    n_sim_conv = 50 if args.quick else 200
    try:
        convergence_df = run_all_convergence(n_sim=n_sim_conv)
    except Exception as e:
        print(f"  Convergence evidence failed: {e}")

    # =========================================================================
    # STEP 5d: Calibration Tables
    # =========================================================================
    print("\n" + "="*60)
    print("CALIBRATION ANALYSIS")
    print("="*60)
    try:
        calib_results = run_calibration_analysis()
    except Exception as e:
        print(f"  Calibration analysis failed: {e}")

    # =========================================================================
    # STEP 6: Robustness and International Experiments
    # =========================================================================
    print("\n" + "="*60)
    print("SECTION 6: Robustness and International Experiments")
    print("="*60)

    # 1. Endpoint Bias (using loaded GDP data)
    if gdp_results is not None:
        print("\n[Experiments] Endpoint Bias Analysis...")
        y_gdp = gdp_results['GDP_Log'].dropna().values * 100 # Scale to percentage
        dates_gdp = gdp_results.index
        run_endpoint_experiments(y_gdp, dates_gdp)
    
    # 2. Hamilton Experiments (Simulations)
    print("\n[Experiments] Hamilton (2018) Replication...")
    run_hamilton_experiments(n_sims=n_sim_sec2)

    # 3. Robustness Checks (Lambda, Sample Periods)
    print("\n[Experiments] Robustness Checks...")
    run_robustness_experiments()

    # 4. International Evidence
    print("\n[Experiments] International Evidence...")
    run_international_experiments()

    # =========================================================================
    # STEP 7: Generate All Figures
    # =========================================================================
    generate_all_figures()

    # Section-specific publication figures
    print("\n[Figures] Section 5: Structural Models...")
    try:
        generate_section5_figure(base_dir)
    except Exception as e:
        print(f"  Section 5 figure failed: {e}")

    print("\n[Figures] Section 6: GDP Output Gap...")
    try:
        generate_section6_gdp_figure(base_dir)
    except Exception as e:
        print(f"  Section 6 GDP figure failed: {e}")

    print("\n[Figures] Section 6: Taylor Rule...")
    try:
        generate_section6_taylor_figure(base_dir)
    except Exception as e:
        print(f"  Section 6 Taylor figure failed: {e}")

    # GDP and SP500 cycles (from empirical results)
    print("\n[Figures] GDP and SP500 Cycles...")
    plot_gdp_cycles()
    if extended_results:
        plot_sp500_cycles(extended_results)

    # Summary figures (event-study, error decomposition, gap difference)
    print("\n[Figures] Summary Figures...")
    try:
        generate_summary_figures()
    except Exception as e:
        print(f"  Summary figures failed: {e}")

    # =========================================================================
    # COMPLETE
    # =========================================================================
    print("\n" + "="*60)
    print("REPLICATION COMPLETE!")
    print("="*60)
    print("\nOutput locations:")
    print(f"  - Results: {os.path.join(base_dir, 'output', 'results')}")
    print(f"  - Figures: {os.path.join(base_dir, 'output', 'figures')}")
    print("\nTo compile the manuscript, run: pdflatex manuscript/paper.tex")


if __name__ == "__main__":
    main()
