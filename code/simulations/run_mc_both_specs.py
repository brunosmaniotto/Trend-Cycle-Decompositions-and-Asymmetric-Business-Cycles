"""
Run Monte Carlo simulations with two lambda specifications:
  1. Default: L1=600, L2=1600 (practitioner recommendation)
  2. Oracle: DGP-specific optimal lambdas for both L1 and L2

Saves results to output/results/mc_default_lambdas.csv and mc_oracle_lambdas.csv.
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


# All 12 DGPs
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

# Oracle-optimal lambdas from sensitivity analysis (Table 1 in paper)
ORACLE_L1 = {
    '1_RandomWalk': 10, '2_DriftRW': 1000, '3_UCModel': 10,
    '4_FatTailedRW': 10, '4b_AsymmetricRW': 10, '5_SkewedUC': 10,
    '6_PhaseShift': 200, '7_StatPlucking': 600, '8_SharpCrash': 2500,
    '9_GarchRW': 10, '10_AsymGarch': 4000, '11_SkewFatTail': 10
}

ORACLE_L2 = {
    '1_RandomWalk': 10, '2_DriftRW': 4000, '3_UCModel': 10,
    '4_FatTailedRW': 10, '4b_AsymmetricRW': 10, '5_SkewedUC': 10,
    '6_PhaseShift': 4000, '7_StatPlucking': 4000, '8_SharpCrash': 4000,
    '9_GarchRW': 10, '10_AsymGarch': 4000, '11_SkewFatTail': 25
}


def run_mc_for_spec(dgps, l1_lambdas, l2_lambdas, n_sim=500, T=200, spec_name=""):
    """Run MC across all DGPs for a given lambda specification."""
    all_summaries = []

    for name, dgp in dgps.items():
        l1_lamb = l1_lambdas[name]
        l2_lamb = l2_lambdas[name]
        print(f"\n--- [{spec_name}] {name} (L1={l1_lamb}, L2={l2_lamb}, {n_sim} sims) ---")

        filters = {
            'HP-L2': lambda y, lamb=None, _l=l2_lamb: l2_hp_filter(y, lamb=_l),
            'HP-L1': lambda y, lamb=None, _l=l1_lamb: l1_hp_filter(y, lamb=_l),
            'Hamilton-L2': l2_hamilton_filter,
            'Hamilton-L1': l1_hamilton_filter
        }

        summary, _ = run_monte_carlo_experiment(dgp, filters, T=T, n_sim=n_sim, lamb=1600)
        summary['DGP'] = name
        summary['L1_Lambda'] = l1_lamb
        summary['L2_Lambda'] = l2_lamb
        all_summaries.append(summary)
        print(summary[['Filter', 'Bias', 'RMSE', 'Cycle Skewness']])

    return pd.concat(all_summaries, ignore_index=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick mode (50 sims)')
    args = parser.parse_args()

    n_sim = 50 if args.quick else 500
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Specification 1: Default lambdas (practitioner recommendation)
    print("\n" + "=" * 60)
    print(f"SPECIFICATION 1: Default Lambdas (L1=600, L2=1600) — {n_sim} sims")
    print("=" * 60)

    default_l1 = {name: 600 for name in DGPS}
    default_l2 = {name: 1600 for name in DGPS}
    results_default = run_mc_for_spec(DGPS, default_l1, default_l2,
                                       n_sim=n_sim, T=200, spec_name="Default")

    path_default = os.path.join(output_dir, 'mc_default_lambdas.csv')
    results_default.to_csv(path_default, index=False)
    print(f"\nSaved: {path_default}")

    # Specification 2: Oracle-optimal lambdas (fairest comparison)
    print("\n" + "=" * 60)
    print(f"SPECIFICATION 2: Oracle Lambdas (DGP-specific) — {n_sim} sims")
    print("=" * 60)

    results_oracle = run_mc_for_spec(DGPS, ORACLE_L1, ORACLE_L2,
                                      n_sim=n_sim, T=200, spec_name="Oracle")

    path_oracle = os.path.join(output_dir, 'mc_oracle_lambdas.csv')
    results_oracle.to_csv(path_oracle, index=False)
    print(f"\nSaved: {path_oracle}")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (HP-L1 vs HP-L2 RMSE)")
    print("=" * 60)
    for name in DGPS:
        d_l2 = results_default[(results_default['DGP'] == name) & (results_default['Filter'] == 'HP-L2')]['RMSE'].values[0]
        d_l1 = results_default[(results_default['DGP'] == name) & (results_default['Filter'] == 'HP-L1')]['RMSE'].values[0]
        o_l2 = results_oracle[(results_oracle['DGP'] == name) & (results_oracle['Filter'] == 'HP-L2')]['RMSE'].values[0]
        o_l1 = results_oracle[(results_oracle['DGP'] == name) & (results_oracle['Filter'] == 'HP-L1')]['RMSE'].values[0]
        d_pct = (d_l1 - d_l2) / d_l2 * 100
        o_pct = (o_l1 - o_l2) / o_l2 * 100
        print(f"  {name:20s}  Default: L2={d_l2:.3f} L1={d_l1:.3f} ({d_pct:+.1f}%)  |  Oracle: L2={o_l2:.3f} L1={o_l1:.3f} ({o_pct:+.1f}%)")

    print("\nDone.")


if __name__ == '__main__':
    main()
