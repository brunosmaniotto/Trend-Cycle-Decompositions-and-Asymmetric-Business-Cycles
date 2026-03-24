import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter
from filters.hamilton import l1_hamilton_filter, l2_hamilton_filter
from config import LAMBDA_L1_EMPIRICAL, LAMBDA_L2_STANDARD, HAC_MAXLAGS

def calculate_taylor_rule(inflation, output_gap, r_star=2.0, pi_star=2.0, alpha=0.5, beta=0.5):
    """
    Calculate Taylor rule interest rate.
    i = r* + pi + alpha(pi - pi*) + beta(y - y*)
    """
    return r_star + inflation + alpha * (inflation - pi_star) + beta * output_gap

def estimate_phillips_curve(inflation, gap, lagged_inflation):
    """
    Estimate Phillips curve parameters: pi_t = alpha + beta*gap_t + gamma*pi_{t-1} + epsilon_t

    Uses Newey-West (HAC) standard errors to account for serial correlation
    in macro data (maxlags determined by config.HAC_MAXLAGS).
    """
    X = np.column_stack([gap, lagged_inflation])
    y = inflation

    # Remove NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    if len(y) < 10:
        return {'beta': np.nan, 'r2': np.nan, 'se_beta': np.nan, 'alpha': np.nan, 'gamma': np.nan, 'n_obs': len(y)}

    # Add intercept
    X_with_const = sm.add_constant(X)

    # OLS with Newey-West HAC standard errors
    ols_model = sm.OLS(y, X_with_const)
    results = ols_model.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_MAXLAGS})

    return {
        'alpha': results.params[0],
        'beta': results.params[1],
        'gamma': results.params[2],
        'se_beta': results.bse[1],
        'r2': results.rsquared,
        'n_obs': int(results.nobs)
    }

def run_policy_analysis(data_path='../../data/processed/fred_data_processed.csv'):
    """
    Run full policy analysis: Taylor Rule and Phillips Curve.
    """
    data_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
    if not os.path.exists(data_full_path):
        print(f"Data file not found for policy analysis: {data_full_path}")
        return None, None
        
    df = pd.read_csv(data_full_path, index_col=0, parse_dates=True)
    
    # Require inflation and unemployment data
    # Calculate inflation from CPI (if available)
    if 'CPILFESL_log' in df.columns:
        if 'CPILFESL_growth' in df.columns:
            inflation = df['CPILFESL_growth']
        else: # Fallback if growth is not precalculated
            inflation = df['CPILFESL_log'].diff() * 400
    else:
        print("CPI data missing. Skipping policy analysis.")
        return None, None # Return None if data missing

    # GDP Output Gaps
    gdp_log = df['GDPC1_log'].values
    
    # Apply filters
    trend_hp_l2, cycle_hp_l2 = l2_hp_filter(gdp_log, lamb=LAMBDA_L2_STANDARD)
    trend_hp_l1, cycle_hp_l1 = l1_hp_filter(gdp_log, lamb=LAMBDA_L1_EMPIRICAL)
    
    # Percent gap
    gap_l2 = cycle_hp_l2 * 100
    gap_l1 = cycle_hp_l1 * 100
    
    # Taylor Rule Analysis
    taylor_l2 = calculate_taylor_rule(inflation, gap_l2)
    taylor_l1 = calculate_taylor_rule(inflation, gap_l1)
    
    df['Taylor_L2'] = taylor_l2
    df['Taylor_L1'] = taylor_l1
    df['Gap_L2'] = gap_l2
    df['Gap_L1'] = gap_l1
    
    # Phillips Curve Analysis
    lagged_inf = inflation.shift(1)
    
    pc_l2_results = estimate_phillips_curve(inflation.values, gap_l2, lagged_inf.values)
    pc_l1_results = estimate_phillips_curve(inflation.values, gap_l1, lagged_inf.values)
    
    print("\nPhillips Curve Results:")
    print(f"L2-HP: Beta = {pc_l2_results['beta']:.4f}, R2 = {pc_l2_results['r2']:.3f}")
    print(f"L1-HP: Beta = {pc_l1_results['beta']:.4f}, R2 = {pc_l1_results['r2']:.3f}")
    
    # Save policy data
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'output', 'results'))
    os.makedirs(output_dir, exist_ok=True)

    policy_paths_file_path = os.path.join(output_dir, 'policy_paths.csv')
    phillips_curve_file_path = os.path.join(output_dir, 'phillips_curve_results.csv')

    df[['Taylor_L2', 'Taylor_L1', 'Gap_L2', 'Gap_L1']].to_csv(policy_paths_file_path, index=True)
    print(f"Saved: {policy_paths_file_path}")

    pc_df = pd.DataFrame([pc_l2_results, pc_l1_results], index=['L2-HP', 'L1-HP'])
    pc_df.to_csv(phillips_curve_file_path, index=True)
    print(f"Saved: {phillips_curve_file_path}")

    return df, pc_l2_results, pc_l1_results


def compute_taylor_rule_counterfactuals(data_path='../../data/processed/fred_data_processed.csv'):
    """
    Compute Taylor Rule counterfactuals for three recession episodes using
    L1-HP and L2-HP output gaps.

    For each episode (Volcker 1979-83, GFC 2007-11, COVID 2019-22), computes
    the prescribed Taylor Rule rate under both L1 and L2 gap measures and
    compares them against the actual federal funds rate.

    Returns a dict with per-episode DataFrames and a summary DataFrame of
    max absolute divergence between L1 and L2 Taylor prescriptions.
    """
    # --- Load data ---
    data_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
    if not os.path.exists(data_full_path):
        print(f"Data file not found: {data_full_path}")
        return None

    df = pd.read_csv(data_full_path, index_col=0, parse_dates=True)

    # --- Inflation ---
    if 'CPILFESL_growth' in df.columns:
        inflation = df['CPILFESL_growth']
    elif 'CPILFESL_log' in df.columns:
        inflation = df['CPILFESL_log'].diff() * 400
    else:
        print("CPI data missing. Cannot compute Taylor counterfactuals.")
        return None

    # --- Federal funds rate ---
    if 'DFF' in df.columns:
        fed_funds = df['DFF']
    else:
        print("Federal funds rate (DFF) missing. Cannot compute Taylor counterfactuals.")
        return None

    # --- Output gaps ---
    gdp_log = df['GDPC1_log'].values

    trend_l2, cycle_l2 = l2_hp_filter(gdp_log, lamb=LAMBDA_L2_STANDARD)
    trend_l1, cycle_l1 = l1_hp_filter(gdp_log, lamb=LAMBDA_L1_EMPIRICAL)

    gap_l2 = cycle_l2 * 100  # percent
    gap_l1 = cycle_l1 * 100

    # --- Taylor Rule prescriptions (full sample) ---
    taylor_l2 = calculate_taylor_rule(inflation, gap_l2)
    taylor_l1 = calculate_taylor_rule(inflation, gap_l1)

    # Build a working DataFrame aligned to the original index
    work = pd.DataFrame({
        'Actual_FFR': fed_funds,
        'Taylor_L1': taylor_l1,
        'Taylor_L2': taylor_l2,
        'Gap_L1': gap_l1,
        'Gap_L2': gap_l2,
        'Inflation': inflation,
    }, index=df.index)

    # --- Episode definitions ---
    episodes = {
        'Volcker': ('1979-07-01', '1983-12-31'),
        'GFC':     ('2007-01-01', '2011-12-31'),
        'COVID':   ('2019-10-01', '2022-12-31'),
    }

    episode_results = {}
    summary_rows = []

    for name, (start, end) in episodes.items():
        ep = work.loc[start:end].dropna(subset=['Taylor_L1', 'Taylor_L2', 'Actual_FFR'])
        episode_results[name] = ep

        max_divergence = np.abs(ep['Taylor_L1'] - ep['Taylor_L2']).max()
        summary_rows.append({
            'Episode': name,
            'Start': start,
            'End': end,
            'Max_Abs_Divergence_L1_L2': max_divergence,
            'N_Quarters': len(ep),
        })

    summary_df = pd.DataFrame(summary_rows)

    # --- Figure: 3-panel plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, (name, (start, end)) in zip(axes, episodes.items()):
        ep = episode_results[name]
        if ep.empty:
            ax.set_title(f"{name} (no data)")
            continue

        ax.plot(ep.index, ep['Actual_FFR'], color='black', linewidth=1.5, label='Actual FFR')
        ax.plot(ep.index, ep['Taylor_L1'], color='tab:red', linewidth=1.5,
                linestyle='--', label='Taylor (L1 gap)')
        ax.plot(ep.index, ep['Taylor_L2'], color='tab:blue', linewidth=1.5,
                linestyle='-.', label='Taylor (L2 gap)')

        ax.set_title(name, fontsize=13)
        ax.set_ylabel('Interest Rate (%)')
        ax.legend(fontsize=8, loc='best')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Taylor Rule Counterfactuals: L1-HP vs L2-HP Output Gaps', fontsize=14, y=1.02)
    fig.tight_layout()

    # --- Save outputs ---
    script_dir = os.path.dirname(__file__)
    fig_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'output', 'figures'))
    res_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'output', 'results'))
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    fig_path = os.path.join(fig_dir, 'taylor_counterfactuals.png')
    csv_path = os.path.join(res_dir, 'taylor_counterfactuals.csv')

    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

    # Combine episode DataFrames for CSV output
    csv_parts = []
    for name, ep in episode_results.items():
        part = ep.copy()
        part['Episode'] = name
        csv_parts.append(part)

    if csv_parts:
        csv_df = pd.concat(csv_parts)
        csv_df.to_csv(csv_path, index=True)
        print(f"Saved CSV:    {csv_path}")

    # Print summary
    print("\nTaylor Rule Counterfactual Summary")
    print("=" * 55)
    for _, row in summary_df.iterrows():
        print(f"  {row['Episode']:>8s}: max |Taylor_L1 - Taylor_L2| = "
              f"{row['Max_Abs_Divergence_L1_L2']:.2f} pp  "
              f"({row['N_Quarters']} quarters)")

    return {
        'episodes': episode_results,
        'summary': summary_df,
        'fig_path': fig_path,
        'csv_path': csv_path,
    }