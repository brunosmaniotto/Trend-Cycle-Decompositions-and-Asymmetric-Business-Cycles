import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from scipy import stats
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter

class LambdaSelection:
    """
    Methods for data-driven selection of smoothing parameter
    """
    
    @staticmethod
    def cross_validation(y, lambda_grid, filter_func, n_splits=5):
        """
        Time series cross-validation for lambda selection
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        y = np.array(y)
        
        for lamb in lambda_grid:
            fold_errors = []
            
            for train_idx, test_idx in tscv.split(y):
                # Skip if test set too small
                if len(test_idx) < 5:
                    continue
                    
                y_train = y[train_idx]
                y_test = y[test_idx]
                
                # Fit on training data
                try:
                    trend_train, _ = filter_func(y_train, lamb=lamb)
                    
                    # Extrapolate trend to test period (simple linear)
                    # Use last two points for extrapolation
                    if len(trend_train) >= 2:
                        slope = trend_train[-1] - trend_train[-2]
                        n_test = len(y_test)
                        trend_test = trend_train[-1] + slope * np.arange(1, n_test + 1)
                        
                        # Calculate prediction error
                        error = np.mean((y_test - trend_test)**2)
                        fold_errors.append(error)
                except Exception:
                    pass

            if fold_errors:
                cv_scores.append({
                    'lambda': lamb,
                    'cv_error': np.mean(fold_errors),
                    'cv_std': np.std(fold_errors)
                })
        
        return pd.DataFrame(cv_scores)
    
    @staticmethod
    def find_equivalent_lambdas(y, base_lambda_l2=1600):
        """
        Find L1 lambda that produces similar smoothness (std of 2nd diff) to L2
        """
        # Get L2 trend smoothness
        trend_l2, _ = l2_hp_filter(y, lamb=base_lambda_l2)
        smoothness_l2 = np.std(np.diff(np.diff(trend_l2)))
        
        # Search for matching L1 lambda
        lambda_search = np.logspace(1, 4, 20)
        smoothness_diff = []
        
        for lamb in lambda_search:
            try:
                trend_l1, _ = l1_hp_filter(y, lamb=lamb)
                smoothness_l1 = np.std(np.diff(np.diff(trend_l1)))
                smoothness_diff.append(abs(smoothness_l1 - smoothness_l2))
            except Exception:
                smoothness_diff.append(np.inf)
        
        # Find best match
        if len(smoothness_diff) > 0:
            best_idx = np.argmin(smoothness_diff)
            equivalent_lambda_l1 = lambda_search[best_idx]
            return equivalent_lambda_l1
        else:
            return np.nan

def run_sensitivity_analysis(y, true_trend=None):
    """
    Run sensitivity analysis on a time series
    """
    lambda_values = np.logspace(1, 4, 10)
    results = []
    
    for lamb in lambda_values:
        # L2
        trend_l2, _ = l2_hp_filter(y, lamb=lamb)
        
        # L1
        try:
            trend_l1, _ = l1_hp_filter(y, lamb=lamb)
            
            res = {
                'lambda': lamb,
                'smoothness_l2': np.std(np.diff(np.diff(trend_l2))),
                'smoothness_l1': np.std(np.diff(np.diff(trend_l1)))
            }
            
            if true_trend is not None:
                res['rmse_l2'] = np.sqrt(np.mean((trend_l2 - true_trend)**2))
                res['rmse_l1'] = np.sqrt(np.mean((trend_l1 - true_trend)**2))
                
            results.append(res)
        except Exception:
            pass

    return pd.DataFrame(results)


# =============================================================================
# Part 3: Monte Carlo Optimal Lambda by DGP
# =============================================================================

def find_optimal_lambda_monte_carlo(dgp, lambda_grid, T=200, n_sim=500,
                                    show_progress=True, filter_func=None):
    """
    Find optimal HP lambda for a given DGP via Monte Carlo simulation.

    For each lambda in the grid, runs n_sim simulations and computes the average
    trend RMSE. The optimal lambda is the one that minimizes mean RMSE.

    Parameters
    ----------
    dgp : DGP instance
        A data generating process with a generate(T, seed) method that returns
        (y, true_trend, true_cycle).
    lambda_grid : array-like
        Grid of lambda values to search over.
    T : int
        Length of each simulated time series.
    n_sim : int
        Number of Monte Carlo simulations per lambda.
    show_progress : bool
        Whether to show a progress bar.
    filter_func : callable, optional
        Filter function with signature f(y, lamb=...) -> (trend, cycle).
        Defaults to l1_hp_filter.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [lambda, mean_rmse, std_rmse, min_rmse, max_rmse]
    """
    if filter_func is None:
        filter_func = l1_hp_filter

    results = []

    iterator = tqdm(lambda_grid, desc="Lambda search") if show_progress else lambda_grid

    for lamb in iterator:
        rmse_values = []

        for sim in range(n_sim):
            try:
                y, true_trend, true_cycle = dgp.generate(T, seed=sim)
                trend_hat, _ = filter_func(y, lamb=lamb)
                rmse = np.sqrt(np.mean((trend_hat - true_trend)**2))
                rmse_values.append(rmse)
            except Exception:
                # Skip failed simulations
                continue

        if rmse_values:
            results.append({
                'lambda': lamb,
                'mean_rmse': np.nanmean(rmse_values),
                'std_rmse': np.nanstd(rmse_values),
                'min_rmse': np.nanmin(rmse_values),
                'max_rmse': np.nanmax(rmse_values),
                'n_valid': len(rmse_values)
            })

    return pd.DataFrame(results)


def run_lambda_selection_experiments(output_dir=None, n_sim=500, T=200):
    """
    Run comprehensive lambda selection analysis across all 12 DGPs.

    For each DGP, finds the optimal L1-HP lambda that minimizes trend RMSE.
    Generates a summary table and RMSE curves figure.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save results. If None, uses default output directory.
    n_sim : int
        Number of Monte Carlo simulations per lambda per DGP.
    T : int
        Length of each simulated time series.

    Returns
    -------
    pd.DataFrame
        Summary table with optimal lambda for each DGP.
    """
    # Import DGPs
    from models.dgps import (
        StatisticalPluckingDGP, RandomWalkDGP, UCModelDGP,
        FatTailedRWDGP, SkewedUCDGP, DriftRandomWalkDGP, AsymmetricRWDGP,
        PhaseShiftDGP, SharpCrashDGP, GarchRWDGP, AsymmetricGarchDGP, SkewedFatTailDGP
    )

    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
    results_dir = os.path.join(output_dir, 'results')
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

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

    # Lambda grid to search
    lambda_grid = [10, 25, 50, 100, 200, 400, 600, 800, 1000, 1600, 2500, 4000]

    # Store results for each DGP
    all_curves = {}
    summary_rows = []

    print("\n" + "="*60)
    print("LAMBDA SELECTION: Monte Carlo Optimal Lambda by DGP")
    print("="*60)
    print(f"Settings: T={T}, n_sim={n_sim}, lambda_grid={lambda_grid}")

    for name, dgp in dgps.items():
        print(f"\n--- Processing {name} ---")

        # Run Monte Carlo search
        curve_df = find_optimal_lambda_monte_carlo(
            dgp, lambda_grid, T=T, n_sim=n_sim, show_progress=True
        )
        all_curves[name] = curve_df

        # Find optimal lambda
        if not curve_df.empty:
            opt_idx = curve_df['mean_rmse'].idxmin()
            opt_lambda = curve_df.loc[opt_idx, 'lambda']
            opt_rmse = curve_df.loc[opt_idx, 'mean_rmse']

            # Get RMSE at reference lambdas
            rmse_600 = curve_df[curve_df['lambda'] == 600]['mean_rmse'].values
            rmse_600 = rmse_600[0] if len(rmse_600) > 0 else np.nan

            rmse_1600 = curve_df[curve_df['lambda'] == 1600]['mean_rmse'].values
            rmse_1600 = rmse_1600[0] if len(rmse_1600) > 0 else np.nan

            summary_rows.append({
                'DGP': name,
                'Optimal_Lambda': opt_lambda,
                'RMSE_at_Optimal': opt_rmse,
                'RMSE_at_600': rmse_600,
                'RMSE_at_1600': rmse_1600,
                'Pct_Loss_600': 100 * (rmse_600 - opt_rmse) / opt_rmse if opt_rmse > 0 else np.nan,
                'Pct_Loss_1600': 100 * (rmse_1600 - opt_rmse) / opt_rmse if opt_rmse > 0 else np.nan
            })

            print(f"  Optimal lambda: {opt_lambda}, RMSE: {opt_rmse:.4f}")
            print(f"  RMSE at 600: {rmse_600:.4f}, RMSE at 1600: {rmse_1600:.4f}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Save summary table
    summary_df.to_csv(os.path.join(results_dir, 'optimal_lambda_by_dgp.csv'), index=False)
    print(f"\nSummary saved to {results_dir}/optimal_lambda_by_dgp.csv")

    # Save all RMSE curves
    curves_list = []
    for name, df in all_curves.items():
        df_copy = df.copy()
        df_copy['DGP'] = name
        curves_list.append(df_copy)
    all_curves_df = pd.concat(curves_list, ignore_index=True)
    all_curves_df.to_csv(os.path.join(results_dir, 'lambda_rmse_curves.csv'), index=False)

    # Generate figure
    _plot_lambda_rmse_curves(all_curves, figures_dir)

    # Print summary table
    print("\n" + "="*60)
    print("OPTIMAL LAMBDA SUMMARY BY DGP")
    print("="*60)
    print(summary_df.to_string(index=False))

    return summary_df, all_curves


def run_l2_lambda_selection_experiments(output_dir=None, n_sim=500, T=200):
    """
    Run L2-HP oracle lambda search across all 12 DGPs.

    Mirrors run_lambda_selection_experiments but uses l2_hp_filter.
    For each DGP, finds the lambda that minimizes trend RMSE under L2.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save results. If None, uses default output directory.
    n_sim : int
        Number of Monte Carlo simulations per lambda per DGP.
    T : int
        Length of each simulated time series.

    Returns
    -------
    pd.DataFrame
        Summary table with optimal L2 lambda for each DGP.
    """
    from models.dgps import (
        StatisticalPluckingDGP, RandomWalkDGP, UCModelDGP,
        FatTailedRWDGP, SkewedUCDGP, DriftRandomWalkDGP, AsymmetricRWDGP,
        PhaseShiftDGP, SharpCrashDGP, GarchRWDGP, AsymmetricGarchDGP, SkewedFatTailDGP
    )

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

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

    lambda_grid = [10, 25, 50, 100, 200, 400, 600, 800, 1000, 1600, 2500, 4000]

    all_curves = {}
    summary_rows = []

    print("\n" + "="*60)
    print("L2-HP ORACLE LAMBDA SELECTION BY DGP")
    print("="*60)
    print(f"Settings: T={T}, n_sim={n_sim}, lambda_grid={lambda_grid}")

    for name, dgp in dgps.items():
        print(f"\n--- Processing {name} (L2) ---")

        curve_df = find_optimal_lambda_monte_carlo(
            dgp, lambda_grid, T=T, n_sim=n_sim, show_progress=True,
            filter_func=l2_hp_filter
        )
        all_curves[name] = curve_df

        if not curve_df.empty:
            opt_idx = curve_df['mean_rmse'].idxmin()
            opt_lambda = curve_df.loc[opt_idx, 'lambda']
            opt_rmse = curve_df.loc[opt_idx, 'mean_rmse']

            rmse_1600 = curve_df[curve_df['lambda'] == 1600]['mean_rmse'].values
            rmse_1600 = rmse_1600[0] if len(rmse_1600) > 0 else np.nan

            summary_rows.append({
                'DGP': name,
                'Optimal_Lambda_L2': opt_lambda,
                'RMSE_at_Optimal': opt_rmse,
                'RMSE_at_1600': rmse_1600,
                'Pct_Loss_1600': 100 * (rmse_1600 - opt_rmse) / opt_rmse if opt_rmse > 0 else np.nan
            })

            print(f"  Optimal lambda: {opt_lambda}, RMSE: {opt_rmse:.4f}")
            print(f"  RMSE at 1600: {rmse_1600:.4f} ({100*(rmse_1600-opt_rmse)/opt_rmse:.1f}% loss)")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(results_dir, 'optimal_lambda_l2_by_dgp.csv'), index=False)
    print(f"\nSummary saved to {results_dir}/optimal_lambda_l2_by_dgp.csv")

    # Save all curves
    curves_list = []
    for name, df in all_curves.items():
        df_copy = df.copy()
        df_copy['DGP'] = name
        curves_list.append(df_copy)
    all_curves_df = pd.concat(curves_list, ignore_index=True)
    all_curves_df.to_csv(os.path.join(results_dir, 'lambda_rmse_curves_l2.csv'), index=False)

    print("\n" + "="*60)
    print("L2-HP OPTIMAL LAMBDA SUMMARY BY DGP")
    print("="*60)
    print(summary_df.to_string(index=False))

    return summary_df


def _plot_lambda_rmse_curves(all_curves, figures_dir):
    """
    Plot RMSE vs lambda curves for key DGPs.
    """
    import matplotlib.pyplot as plt

    # Key DGPs to highlight (asymmetric + baseline)
    key_dgps = ['3_UCModel', '5_SkewedUC', '7_StatPlucking', '8_SharpCrash']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, dgp_name in zip(axes, key_dgps):
        if dgp_name in all_curves:
            df = all_curves[dgp_name]

            ax.plot(df['lambda'], df['mean_rmse'], 'b-o', linewidth=2, markersize=6)
            ax.fill_between(
                df['lambda'],
                df['mean_rmse'] - df['std_rmse'],
                df['mean_rmse'] + df['std_rmse'],
                alpha=0.2
            )

            # Mark optimal
            opt_idx = df['mean_rmse'].idxmin()
            opt_lambda = df.loc[opt_idx, 'lambda']
            opt_rmse = df.loc[opt_idx, 'mean_rmse']
            ax.axvline(x=opt_lambda, color='green', linestyle='--',
                      label=f'Optimal: λ={opt_lambda}')

            # Mark reference lambdas
            ax.axvline(x=600, color='orange', linestyle=':', label='λ=600')
            ax.axvline(x=1600, color='red', linestyle=':', label='λ=1600')

            ax.set_xscale('log')
            ax.set_xlabel('Lambda (log scale)')
            ax.set_ylabel('Mean Trend RMSE')
            ax.set_title(dgp_name.replace('_', ' '))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'lambda_rmse_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {figures_dir}/lambda_rmse_curves.png")


# =============================================================================
# Part 2: Empirical Sensitivity Analysis
# =============================================================================

def run_lambda_sensitivity_empirical(data_path=None, output_dir=None):
    """
    Test sensitivity of key empirical findings to lambda choice.

    Shows that cycle skewness and recession trough depths are robust
    across λ = 400, 600, 800, 1000.

    Parameters
    ----------
    data_path : str, optional
        Path to processed FRED data CSV.
    output_dir : str, optional
        Directory to save results.

    Returns
    -------
    pd.DataFrame
        Sensitivity results across lambda values.
    """
    # Set up paths
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'processed',
            'fred_data_processed.csv'
        )
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    try:
        data = pd.read_csv(data_path, parse_dates=['DATE'], index_col='DATE')
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Please run data download first.")
        return None

    # Get GDP series (log levels)
    if 'GDPC1_log' not in data.columns:
        print("GDP log series not found in data.")
        return None

    gdp = data['GDPC1_log'].dropna().values
    dates = data['GDPC1_log'].dropna().index

    # Lambda values to test
    lambda_values = [400, 600, 800, 1000]

    # Key recession trough dates (quarterly)
    recession_troughs = {
        '1981-82': '1982-12-01',
        '2007-09': '2009-06-01',
        '2020': '2020-06-01'
    }

    results = []

    print("\n" + "="*60)
    print("EMPIRICAL SENSITIVITY ANALYSIS")
    print("="*60)

    for lamb in lambda_values:
        print(f"\n--- Lambda = {lamb} ---")

        try:
            trend, cycle = l1_hp_filter(gdp, lamb=lamb)

            # Compute skewness
            skewness = stats.skew(cycle)

            # Find trough gaps
            trough_gaps = {}
            for name, date_str in recession_troughs.items():
                try:
                    trough_date = pd.Timestamp(date_str)
                    # Find closest date in index
                    idx = dates.get_indexer([trough_date], method='nearest')[0]
                    if idx >= 0 and idx < len(cycle):
                        trough_gaps[name] = cycle[idx] * 100  # Convert to percentage
                except Exception:
                    trough_gaps[name] = np.nan

            result = {
                'lambda': lamb,
                'cycle_skewness': skewness,
                'trough_1981_82': trough_gaps.get('1981-82', np.nan),
                'trough_2007_09': trough_gaps.get('2007-09', np.nan),
                'trough_2020': trough_gaps.get('2020', np.nan),
                'cycle_std': np.std(cycle) * 100
            }
            results.append(result)

            print(f"  Skewness: {skewness:.3f}")
            print(f"  Cycle std: {result['cycle_std']:.2f}%")
            for name, gap in trough_gaps.items():
                if not np.isnan(gap):
                    print(f"  {name} trough gap: {gap:.2f}%")

        except Exception as e:
            print(f"  Error: {e}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv(os.path.join(output_dir, 'lambda_sensitivity_empirical.csv'), index=False)
    print(f"\nResults saved to {output_dir}/lambda_sensitivity_empirical.csv")

    # Print summary table
    print("\n" + "="*60)
    print("EMPIRICAL SENSITIVITY SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))

    return results_df


# =============================================================================
# Part 1: Equivalent Smoothness Analysis
# =============================================================================

def run_equivalent_smoothness_analysis(data_path=None):
    """
    Find the L1-HP lambda that produces equivalent smoothness to L2-HP at λ=1600.

    Uses US GDP data to show that λ≈600 emerges from principled matching.

    Parameters
    ----------
    data_path : str, optional
        Path to processed FRED data CSV.

    Returns
    -------
    float
        The equivalent L1-HP lambda value.
    """
    # Set up path
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'processed',
            'fred_data_processed.csv'
        )

    # Load data
    try:
        data = pd.read_csv(data_path, parse_dates=['DATE'], index_col='DATE')
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        return None

    if 'GDPC1_log' not in data.columns:
        print("GDP log series not found in data.")
        return None

    gdp = data['GDPC1_log'].dropna().values

    print("\n" + "="*60)
    print("EQUIVALENT SMOOTHNESS ANALYSIS")
    print("="*60)

    # Find equivalent lambda
    selector = LambdaSelection()
    equivalent_lambda = selector.find_equivalent_lambdas(gdp, base_lambda_l2=1600)

    print(f"\nBase L2-HP lambda: 1600")
    print(f"Equivalent L1-HP lambda: {equivalent_lambda:.1f}")

    # Show smoothness comparison
    trend_l2, _ = l2_hp_filter(gdp, lamb=1600)
    smoothness_l2 = np.std(np.diff(np.diff(trend_l2)))

    trend_l1_eq, _ = l1_hp_filter(gdp, lamb=equivalent_lambda)
    smoothness_l1_eq = np.std(np.diff(np.diff(trend_l1_eq)))

    trend_l1_600, _ = l1_hp_filter(gdp, lamb=600)
    smoothness_l1_600 = np.std(np.diff(np.diff(trend_l1_600)))

    print(f"\nTrend smoothness (std of 2nd diff):")
    print(f"  L2-HP (λ=1600):     {smoothness_l2:.6f}")
    print(f"  L1-HP (λ={equivalent_lambda:.0f}):  {smoothness_l1_eq:.6f}")
    print(f"  L1-HP (λ=600):      {smoothness_l1_600:.6f}")

    return equivalent_lambda


# =============================================================================
# Main execution
# =============================================================================

def run_all_lambda_selection_experiments(n_sim=500, T=200):
    """
    Run complete lambda selection analysis (all three parts).
    """
    print("\n" + "="*70)
    print("COMPLETE LAMBDA SELECTION ANALYSIS")
    print("="*70)

    # Part 1: Equivalent smoothness
    print("\n[PART 1] Equivalent Smoothness Justification")
    equivalent_lambda = run_equivalent_smoothness_analysis()

    # Part 2: Empirical sensitivity
    print("\n[PART 2] Empirical Sensitivity Analysis")
    empirical_df = run_lambda_sensitivity_empirical()

    # Part 3: Monte Carlo optimal lambda (most important)
    print("\n[PART 3] Monte Carlo Optimal Lambda by DGP")
    summary_df, all_curves = run_lambda_selection_experiments(n_sim=n_sim, T=T)

    return {
        'equivalent_lambda': equivalent_lambda,
        'empirical_sensitivity': empirical_df,
        'optimal_lambda_summary': summary_df,
        'rmse_curves': all_curves
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Lambda Selection Analysis')
    parser.add_argument('--quick', action='store_true',
                       help='Quick run with reduced simulations')
    parser.add_argument('--n_sim', type=int, default=500,
                       help='Number of Monte Carlo simulations')
    parser.add_argument('--T', type=int, default=200,
                       help='Time series length')
    args = parser.parse_args()

    if args.quick:
        n_sim = 50
        T = 100
    else:
        n_sim = args.n_sim
        T = args.T

    results = run_all_lambda_selection_experiments(n_sim=n_sim, T=T)
