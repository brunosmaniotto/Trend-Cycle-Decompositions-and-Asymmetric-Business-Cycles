import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os

# Add parent directory to path to import filters
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter
from filters.hamilton import l1_hamilton_filter, l2_hamilton_filter
from config import LAMBDA_L1_EMPIRICAL, LAMBDA_L2_STANDARD

def run_gdp_analysis(data_path='../../data/processed/fred_data_processed.csv'):
    """
    Run filter analysis on Real GDP.
    """
    # Load data
    data_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
    if not os.path.exists(data_full_path):
        print(f"Data file not found for GDP analysis: {data_full_path}")
        return None, None
        
    df = pd.read_csv(data_full_path, index_col=0, parse_dates=True)
    gdp_log = df['GDPC1_log'].values
    dates = df.index
    
    print("Applying filters to Real GDP...")
    
    # Apply filters
    trend_hp_l2, cycle_hp_l2 = l2_hp_filter(gdp_log, lamb=LAMBDA_L2_STANDARD)
    trend_hp_l1, cycle_hp_l1 = l1_hp_filter(gdp_log, lamb=LAMBDA_L1_EMPIRICAL)
    
    trend_ham_l2, cycle_ham_l2 = l2_hamilton_filter(gdp_log)
    trend_ham_l1, cycle_ham_l1 = l1_hamilton_filter(gdp_log)
    
    # Collect results
    results = pd.DataFrame({
        'Date': dates,
        'GDP_Log': gdp_log,
        'Cycle_HP_L2': cycle_hp_l2,
        'Cycle_HP_L1': cycle_hp_l1,
        'Cycle_Ham_L2': cycle_ham_l2,
        'Cycle_Ham_L1': cycle_ham_l1
    })
    
    # Compute statistics
    stats_df = pd.DataFrame({
        'Filter': ['HP-L2', 'HP-L1', 'Hamilton-L2', 'Hamilton-L1'],
        'Skewness': [
            stats.skew(cycle_hp_l2),
            stats.skew(cycle_hp_l1),
            stats.skew(cycle_ham_l2[~np.isnan(cycle_ham_l2)]),
            stats.skew(cycle_ham_l1[~np.isnan(cycle_ham_l1)])
        ],
        'Kurtosis': [
            stats.kurtosis(cycle_hp_l2),
            stats.kurtosis(cycle_hp_l1),
            stats.kurtosis(cycle_ham_l2[~np.isnan(cycle_ham_l2)]),
            stats.kurtosis(cycle_ham_l1[~np.isnan(cycle_ham_l1)])
        ]
    })
    
    print("\nFilter Statistics:")
    print(stats_df)
    
    # Save results
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(script_dir, '../../output/results'))
    os.makedirs(output_dir, exist_ok=True)
    
    results_file_path = os.path.join(output_dir, 'gdp_filter_results.csv')
    stats_file_path = os.path.join(output_dir, 'gdp_filter_stats.csv')

    results.to_csv(results_file_path, index=False)
    stats_df.to_csv(stats_file_path, index=False)
    print(f"Saved: {results_file_path}")
    print(f"Saved: {stats_file_path}")
    
    return results, stats_df