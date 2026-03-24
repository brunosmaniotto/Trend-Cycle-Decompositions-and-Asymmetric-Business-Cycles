import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Ensure correct path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dgps import StatisticalPluckingDGP
from models.structural import FinancialAcceleratorModel
from utils.bootstrap import bootstrap_l1_hp
from data_processing.downloader import download_fred_data
from data_processing.preprocessor import preprocess_fred_data

def plot_bootstrap(ax, time, y, trend, lower, upper, title):
    ax.plot(time, y, 'k.', alpha=0.2, label='Data', markersize=2)
    ax.plot(time, trend, 'r-', linewidth=1.5, label='L1 Trend')
    ax.fill_between(time, lower, upper, color='red', alpha=0.2, label='90% CI')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

def run_bootstrap_demo():
    print("Running Bootstrap Uncertainty Quantification Demo...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # 1. Statistical Plucking DGP
    print("  - Case 1: Statistical Plucking...")
    dgp = StatisticalPluckingDGP()
    y_pluck, _, _ = dgp.generate(100, seed=42) # Reduced T for speed
    trend_p, low_p, high_p, _ = bootstrap_l1_hp(y_pluck, lamb=600, n_boot=50)
    plot_bootstrap(axes[0], np.arange(100), y_pluck, trend_p, low_p, high_p, 
                   "Statistical Plucking DGP: L1 Trend Uncertainty")
    
    # 2. Financial Accelerator Model
    print("  - Case 2: Financial Accelerator...")
    model = FinancialAcceleratorModel()
    sim_data = model.simulate(T=100)
    y_fin = sim_data['output']
    trend_f, low_f, high_f, _ = bootstrap_l1_hp(y_fin, lamb=600, n_boot=50)
    plot_bootstrap(axes[1], np.arange(100), y_fin, trend_f, low_f, high_f, 
                   "Financial Accelerator Model: L1 Trend Uncertainty")
    
    # 3. US Real GDP
    print("  - Case 3: US Real GDP...")
    # Quick load (assuming data exists from previous runs)
    data_path = os.path.join(os.path.dirname(__file__), '../../data/processed/fred_data_processed.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        if 'GDPC1_log' in df.columns:
            y_gdp = df['GDPC1_log'].values
            dates = df.index
            
            # Run bootstrap on recent history (last 100 qtrs) for speed/visibility
            y_gdp_recent = y_gdp[-100:]
            dates_recent = dates[-100:]
            
            trend_g, low_g, high_g, _ = bootstrap_l1_hp(y_gdp_recent, lamb=600, n_boot=50)
            
            axes[2].plot(dates_recent, y_gdp_recent, 'k.', alpha=0.2, label='Data')
            axes[2].plot(dates_recent, trend_g, 'r-', linewidth=1.5, label='L1 Trend')
            axes[2].fill_between(dates_recent, low_g, high_g, color='red', alpha=0.2, label='90% CI')
            axes[2].set_title("US Real GDP (Recent): L1 Trend Uncertainty")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/figures'))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'bootstrap_uncertainty.png')
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    run_bootstrap_demo()
