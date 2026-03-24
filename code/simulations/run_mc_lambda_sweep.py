"""
Run Monte Carlo simulations sweeping L1 lambda from 600 to 2000 (step 100).
L2 fixed at 1600 throughout. Reports RMSE for HP-L1 and HP-L2 across all 12 DGPs.

Saves results to output/results/mc_lambda_sweep.csv
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dgps import (
    RandomWalkDGP, DriftRandomWalkDGP, UCModelDGP, FatTailedRWDGP,
    AsymmetricRWDGP, SkewedUCDGP, PhaseShiftDGP, StatisticalPluckingDGP,
    SharpCrashDGP, GarchRWDGP, AsymmetricGarchDGP, SkewedFatTailDGP
)
from filters.hp import l1_hp_filter, l2_hp_filter
from filters.hamilton import l1_hamilton_filter, l2_hamilton_filter
from simulations.monte_carlo import run_monte_carlo_experiment


DGPS = {
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

LAMBDA_GRID = list(range(600, 2100, 100))  # 600, 700, ..., 2000


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode (50 sims)')
    args = parser.parse_args()

    n_sim = 50 if args.quick else 500
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'results')
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []

    for l1_lamb in LAMBDA_GRID:
        print(f"\n{'='*60}")
        print(f"L1 Lambda = {l1_lamb} (L2 = 1600) — {n_sim} sims")
        print(f"{'='*60}")

        for name, dgp in DGPS.items():
            print(f"  {name}...", end=" ", flush=True)

            filters = {
                'HP-L2': lambda y, lamb=None: l2_hp_filter(y, lamb=1600),
                'HP-L1': lambda y, lamb=None, _l=l1_lamb: l1_hp_filter(y, lamb=_l),
            }

            summary, _ = run_monte_carlo_experiment(dgp, filters, T=200, n_sim=n_sim, lamb=1600)

            for _, row in summary.iterrows():
                all_rows.append({
                    'L1_Lambda': l1_lamb,
                    'DGP': name,
                    'Filter': row['Filter'],
                    'Bias': row['Bias'],
                    'RMSE': row['RMSE'],
                    'Cycle_Skewness': row['Cycle Skewness'],
                    'Cycle_Std_Dev': row['Cycle Std Dev'],
                })

            l2_rmse = summary[summary['Filter'] == 'HP-L2']['RMSE'].values[0]
            l1_rmse = summary[summary['Filter'] == 'HP-L1']['RMSE'].values[0]
            pct = (l1_rmse - l2_rmse) / l2_rmse * 100
            print(f"L2={l2_rmse:.3f}  L1={l1_rmse:.3f}  ({pct:+.1f}%)")

    results = pd.DataFrame(all_rows)
    path = os.path.join(output_dir, 'mc_lambda_sweep.csv')
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # Print summary: for each lambda, count how many DGPs L1 wins
    print(f"\n{'='*60}")
    print("SUMMARY: L1 wins (RMSE < L2) by lambda")
    print(f"{'='*60}")
    for l1_lamb in LAMBDA_GRID:
        subset = results[results['L1_Lambda'] == l1_lamb]
        wins = 0
        total = 0
        for dgp_name in DGPS:
            l2_rmse = subset[(subset['DGP'] == dgp_name) & (subset['Filter'] == 'HP-L2')]['RMSE'].values[0]
            l1_rmse = subset[(subset['DGP'] == dgp_name) & (subset['Filter'] == 'HP-L1')]['RMSE'].values[0]
            if l1_rmse < l2_rmse:
                wins += 1
            total += 1
        print(f"  Lambda={l1_lamb:5d}: L1 wins {wins}/{total} DGPs")


if __name__ == '__main__':
    main()
