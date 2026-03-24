import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os

# Ensure we can import from the code directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dgps import (
    RandomWalkDGP,
    DriftRandomWalkDGP,
    UCModelDGP,
    FatTailedRWDGP,
    AsymmetricRWDGP,
    SkewedUCDGP,
    SkewedFatTailDGP, # New import
    PhaseShiftDGP,
    StatisticalPluckingDGP,
    SharpCrashDGP,
    GarchRWDGP,
    AsymmetricGarchDGP
)
from filters.hp import l1_hp_filter, l2_hp_filter
from filters.hamilton import l1_hamilton_filter, l2_hamilton_filter
from simulations.monte_carlo import run_monte_carlo_experiment

def main():
    print("Running Section 2 Simulations (Statistical Benchmarks)...")
    
    # Define DGPs
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
        '11_SkewFatTail': SkewedFatTailDGP() # New DGP
    }
    
    # Optimal HP-L1 lambdas identified from previous sensitivity run
    optimal_hp_l1_lambdas = {
        '1_RandomWalk': 400,
        '2_DriftRW': 400,
        '3_UCModel': 10,
        '4_FatTailedRW': 400,
        '4b_AsymmetricRW': 400,
        '5_SkewedUC': 10,
        '6_PhaseShift': 200,
        '7_StatPlucking': 600,
        '8_SharpCrash': 1600,
        '9_GarchRW': 400,
        '10_AsymGarch': 10,
        '11_SkewFatTail': 10 # Default to same as SkewedUC
    }
    
    # Parameters for Monte Carlo
    T = 200 
    n_sim = 500
    
    all_summaries = []
    
    for name, dgp in dgps.items():
        print(f"\n--- Running {name} ---")
        
        # Define filters for this specific DGP run
        filters_for_run = {
            'HP-L2': lambda y, lamb=1600: l2_hp_filter(y, lamb=1600), 
            'HP-L1': lambda y, lamb=optimal_hp_l1_lambdas.get(name, 600): l1_hp_filter(y, lamb=optimal_hp_l1_lambdas.get(name, 600)),
            'Hamilton-L2': l2_hamilton_filter,
            'Hamilton-L1': l1_hamilton_filter
        }

        summary, _ = run_monte_carlo_experiment(dgp, filters_for_run, T=T, n_sim=n_sim, lamb=1600) 
        summary['DGP'] = name
        all_summaries.append(summary)
        print(summary[['Filter', 'Bias', 'RMSE', 'Cycle Skewness']])
        
    # Combine and Save
    full_results = pd.concat(all_summaries, ignore_index=True)
    
    # Define output directory robustly
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(script_dir, '../../output/results'))
    os.makedirs(output_dir, exist_ok=True)
    full_results.to_csv(os.path.join(output_dir, 'section2_sim_results_final.csv'), index=False)
    
    print(f"\nAll simulations complete. Results saved to {output_dir}/section2_sim_results_final.csv")

if __name__ == "__main__":
    main()
