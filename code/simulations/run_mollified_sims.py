import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm

# Ensure correct path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dgps import StatisticalPluckingDGP, FatTailedRWDGP
from filters.hp import l1_hp_filter, l2_hp_filter
from filters.robust_smooth import huber_hp_filter, elastic_hp_filter
from simulations.monte_carlo import run_monte_carlo_experiment

def main():
    print("Running Mollified Filter Simulations (Huber/Elastic)...")
    
    # Select Emblematic DGPs
    dgps = {
        'Plucking': StatisticalPluckingDGP(),
        'FatTailed': FatTailedRWDGP()
    }
    
    results = []
    
    for name, dgp in dgps.items():
        print(f"\n--- Testing {name} ---")
        
        # Define filter suite
        # Accept **kwargs so the MC runner can pass 'lamb' without error
        filters = {
            'HP-L2': lambda y, **kwargs: l2_hp_filter(y, lamb=1600),
            'HP-L1': lambda y, **kwargs: l1_hp_filter(y, lamb=600),
            'Huber-HP (d=1)': lambda y, **kwargs: huber_hp_filter(y, lamb=1000, delta=1.0),
            'Huber-HP (d=0.1)': lambda y, **kwargs: huber_hp_filter(y, lamb=1000, delta=0.1),
            'Elastic-HP (a=0.5)': lambda y, **kwargs: elastic_hp_filter(y, lamb=1000, alpha=0.5)
        }
        
        # Using T=100 and n_sim=50 for speed in this demo, can increase if needed
        summary, _ = run_monte_carlo_experiment(dgp, filters, T=100, n_sim=50, lamb=1600)
        summary['DGP'] = name
        results.append(summary)
        print(summary[['Filter', 'Bias', 'RMSE', 'Cycle Skewness']])

    # Save
    final_df = pd.concat(results, ignore_index=True)
    
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(script_dir, '../../output/results'))
    os.makedirs(output_dir, exist_ok=True)
    
    final_df.to_csv(os.path.join(output_dir, 'mollified_sim_results.csv'), index=False)
    print(f"\nResults saved to {output_dir}/mollified_sim_results.csv")

if __name__ == "__main__":
    main()
