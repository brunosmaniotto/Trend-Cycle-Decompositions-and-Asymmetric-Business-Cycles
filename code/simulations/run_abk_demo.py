import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.asymmetric_bk import asymmetric_bk_filter, standard_bk_filter
from models.dgps import StatisticalPluckingDGP

def run_demo():
    print("Running Asymmetric Baxter-King Demo...")
    
    # Generate Plucking Data
    dgp = StatisticalPluckingDGP()
    y, true_trend, true_cycle = dgp.generate(200, seed=42)
    
    # Apply Standard BK
    bk_cycle = standard_bk_filter(y)
    
    # Apply Asymmetric BK (L1)
    abk_trend, abk_cycle = asymmetric_bk_filter(y, method='L1')
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(true_cycle, 'k--', label='True Plucking Cycle', alpha=0.5)
    plt.plot(bk_cycle, 'b-', label='Standard BK (Symmetric)', alpha=0.7)
    plt.plot(abk_cycle, 'r-', label='Asymmetric BK (Split Basis)', alpha=0.7)
    plt.axhline(0, color='k', linestyle=':', alpha=0.3)
    plt.title('Asymmetric Band-Pass Filtering: Plucking Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/figures'))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'abk_demo.png')
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    run_demo()
