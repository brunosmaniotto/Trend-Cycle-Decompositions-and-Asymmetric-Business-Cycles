"""
Joint (p, lambda) optimization for the Lp-HP filter.
Grid search over p in [1, 2] and lambda in a range.
Finds oracle-optimal (p*, lambda*) for each DGP and structural model.
"""

import os
import numpy as np
import cvxpy as cp
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cvxpy')

from models.structural import (
    PluckingModel, FinancialAcceleratorModel,
    UncertaintyShocksModel, HysteresisModel
)
from models.dgps import (
    StatisticalPluckingDGP, RandomWalkDGP, UCModelDGP, SharpCrashDGP,
    SkewedUCDGP, AsymmetricRWDGP
)


def lp_hp_filter(y, lamb=1600, p=1.5, solver=cp.CLARABEL):
    """HP filter with Lp norm (p between 1 and 2)."""
    y = np.asarray(y, dtype=float).flatten()
    T = len(y)
    tau = cp.Variable(T)

    residuals = y - tau
    second_diff = tau[2:] - 2 * tau[1:-1] + tau[:-2]

    if p == 1.0:
        fit_cost = cp.norm1(residuals)
        smooth_cost = cp.norm1(second_diff)
    elif p == 2.0:
        fit_cost = cp.sum_squares(residuals)
        smooth_cost = cp.sum_squares(second_diff)
    else:
        fit_cost = cp.pnorm(residuals, p) ** p
        smooth_cost = cp.pnorm(second_diff, p) ** p

    objective = cp.Minimize(fit_cost + lamb * smooth_cost)
    problem = cp.Problem(objective)

    try:
        problem.solve(solver=solver, verbose=False)
    except (cp.SolverError, Exception):
        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=10000)
        except Exception:
            return None, None

    if tau.value is None:
        return None, None

    trend = np.array(tau.value).flatten()
    cycle = y - trend
    return trend, cycle


def run_lp_joint_experiment(base_dir, n_sim=50, T=200):
    """Run joint (p, lambda) grid search on DGPs and structural models."""

    print("\n" + "=" * 70)
    print("Lp JOINT EXPERIMENT: Oracle-optimal (p*, lambda*)")
    print("=" * 70)

    out_dir = os.path.join(base_dir, 'output', 'results', 'lp_experiment')
    os.makedirs(out_dir, exist_ok=True)

    P_GRID = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0]
    LAMBDA_GRID = [100, 200, 400, 600, 800, 1000, 1200, 1600, 2000, 2500]

    # DGP experiments
    dgps = {
        'StatPlucking': StatisticalPluckingDGP(),
        'RandomWalk': RandomWalkDGP(),
        'UCModel': UCModelDGP(),
        'SharpCrash': SharpCrashDGP(),
        'SkewedUC': SkewedUCDGP(),
        'AsymRW': AsymmetricRWDGP(),
    }

    print(f"\n  p grid: {P_GRID}")
    print(f"  lambda grid: {LAMBDA_GRID}")
    print(f"  N_SIM={n_sim}, T={T}")

    # Pre-generate DGP data
    print("\n  Pre-generating DGP data...")
    dgp_data = {}
    for dgp_name, dgp in dgps.items():
        dgp_data[dgp_name] = []
        for sim in range(n_sim):
            y, true_trend, true_cycle = dgp.generate(T=T, seed=sim * 100 + 42)
            dgp_data[dgp_name].append((y, true_trend, true_cycle))

    dgp_results = []
    for dgp_name in dgps:
        print(f"\n  {dgp_name}:")
        for p in P_GRID:
            for lamb in LAMBDA_GRID:
                rmses = []
                for sim in range(n_sim):
                    y, true_trend, true_cycle = dgp_data[dgp_name][sim]
                    trend, cycle = lp_hp_filter(y, lamb=lamb, p=p)
                    if trend is None:
                        continue
                    rmse = np.sqrt(np.mean((cycle - true_cycle) ** 2))
                    rmses.append(rmse)

                mean_rmse = np.mean(rmses) if rmses else np.nan
                dgp_results.append({
                    'DGP': dgp_name, 'p': p, 'lambda': lamb,
                    'RMSE': mean_rmse, 'n_valid': len(rmses)
                })

            sub = [r for r in dgp_results if r['DGP'] == dgp_name and r['p'] == p]
            best = min(sub, key=lambda x: x['RMSE'])
            print(f"    p={p:.1f}: best lambda={best['lambda']:5d}  "
                  f"RMSE={best['RMSE']:.4f}")

    df_dgp = pd.DataFrame(dgp_results)
    df_dgp.to_csv(os.path.join(out_dir, 'lp_joint_dgp_results.csv'), index=False)

    print("\n  Oracle optimal (p*, lambda*) by DGP:")
    print(f"  {'DGP':15s} {'p*':>5s} {'lambda*':>8s} {'RMSE':>8s}")
    print("  " + "-" * 40)
    for dgp_name in dgps:
        sub = df_dgp[df_dgp['DGP'] == dgp_name]
        best = sub.loc[sub['RMSE'].idxmin()]
        print(f"  {dgp_name:15s} {best['p']:5.1f} {best['lambda']:8.0f} "
              f"{best['RMSE']:8.4f}")

    # Structural model experiments
    print("\n" + "=" * 70)
    print("  STRUCTURAL MODELS")
    print("=" * 70)

    struct_models = {
        'Plucking': PluckingModel(),
        'FinAccel': FinancialAcceleratorModel(),
        'Uncertainty': UncertaintyShocksModel(),
        'Hysteresis': HysteresisModel(),
    }

    N_REP = min(20, n_sim)

    print(f"\n  Pre-generating structural model data (N_REP={N_REP})...")
    struct_data = {}
    for model_name, model in struct_models.items():
        struct_data[model_name] = []
        for seed in range(N_REP):
            np.random.seed(seed)
            sim = model.simulate(T=T, burn_in=50)
            struct_data[model_name].append((sim['output'], sim['cycle']))

    struct_results = []
    for model_name in struct_models:
        print(f"\n  {model_name}:")
        for p in P_GRID:
            for lamb in LAMBDA_GRID:
                rmses = []
                for rep in range(N_REP):
                    y, true_cycle = struct_data[model_name][rep]
                    trend, cycle = lp_hp_filter(y, lamb=lamb, p=p)
                    if trend is None:
                        continue
                    rmse = np.sqrt(np.mean((cycle - true_cycle) ** 2))
                    rmses.append(rmse)

                mean_rmse = np.mean(rmses) if rmses else np.nan
                struct_results.append({
                    'Model': model_name, 'p': p, 'lambda': lamb,
                    'RMSE': mean_rmse, 'n_valid': len(rmses)
                })

            sub = [r for r in struct_results
                   if r['Model'] == model_name and r['p'] == p]
            best = min(sub, key=lambda x: x['RMSE'])
            print(f"    p={p:.1f}: best lambda={best['lambda']:5d}  "
                  f"RMSE={best['RMSE']:.4f}")

    df_struct = pd.DataFrame(struct_results)
    df_struct.to_csv(os.path.join(out_dir, 'lp_joint_structural_results.csv'),
                      index=False)

    print("\n  Oracle optimal (p*, lambda*) by model:")
    print(f"  {'Model':15s} {'p*':>5s} {'lambda*':>8s} {'RMSE':>8s}")
    print("  " + "-" * 40)
    for model_name in struct_models:
        sub = df_struct[df_struct['Model'] == model_name]
        best = sub.loc[sub['RMSE'].idxmin()]
        print(f"  {model_name:15s} {best['p']:5.1f} {best['lambda']:8.0f} "
              f"{best['RMSE']:8.4f}")

    print(f"\n  All Lp results saved to: {out_dir}")
