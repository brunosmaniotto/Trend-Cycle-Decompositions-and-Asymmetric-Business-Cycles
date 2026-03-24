import numpy as np
import pandas as pd
from tqdm import tqdm

def run_monte_carlo_experiment(dgp, filters, T=200, n_sim=500, lamb=1600):
    """
    Run Monte Carlo experiment for a given DGP and set of filters
    
    Parameters:
    -----------
    dgp : DGP object
        Data generating process
    filters : dict
        Dictionary of filter functions {name: function}
    T : int
        Sample size
    n_sim : int
        Number of simulations
    lamb : float
        HP filter smoothing parameter
    
    Returns:
    --------
    summary : pd.DataFrame
        Aggregated performance metrics
    results : dict
        Raw results for distribution analysis
    """
    
    # Storage for results
    results = {name: {'bias': [], 'rmse': [], 'cycle_skew': [], 'cycle_std': []} 
               for name in filters.keys()}
    
    # Run simulations
    for sim in tqdm(range(n_sim), desc=f"Monte Carlo ({dgp.__class__.__name__})"):
        # Generate data
        y, true_trend, true_cycle = dgp.generate(T, seed=sim)
        
        # Apply each filter
        for name, filter_func in filters.items():
            try:
                # Check signature to handle lambda correctly
                # Simple heuristic: if 'hp' is in the name, pass lambda
                if 'hp' in name.lower():
                    trend_hat, cycle_hat = filter_func(y, lamb=lamb)
                else:
                    trend_hat, cycle_hat = filter_func(y)
                
                # Compute metrics (ignoring NaN values)
                valid = ~np.isnan(trend_hat)
                
                if valid.sum() > 0:
                    bias = np.mean(trend_hat[valid] - true_trend[valid])
                    rmse = np.sqrt(np.mean((trend_hat[valid] - true_trend[valid])**2))
                    
                    valid_cycle = ~np.isnan(cycle_hat)
                    if valid_cycle.sum() > 3:  # Need at least 3 points for skewness
                        cycle_skew = pd.Series(cycle_hat[valid_cycle]).skew()
                        cycle_std = np.std(cycle_hat[valid_cycle])
                    else:
                        cycle_skew = np.nan
                        cycle_std = np.nan
                    
                    results[name]['bias'].append(bias)
                    results[name]['rmse'].append(rmse)
                    results[name]['cycle_skew'].append(cycle_skew)
                    results[name]['cycle_std'].append(cycle_std)
                else:
                    # Filter failed to produce valid output
                    for metric in results[name].keys():
                        results[name][metric].append(np.nan)
            except Exception as e:
                # Filter crashed
                # print(f"Filter {name} failed: {e}")
                for metric in results[name].keys():
                    results[name][metric].append(np.nan)
    
    # Aggregate results
    summary = pd.DataFrame()
    for name in filters.keys():
        row = pd.Series({
            'Filter': name,
            'Bias': np.nanmean(results[name]['bias']),
            'RMSE': np.nanmean(results[name]['rmse']),
            'Cycle Skewness': np.nanmean(results[name]['cycle_skew']),
            'Cycle Std Dev': np.nanmean(results[name]['cycle_std'])
        })
        summary = pd.concat([summary, row.to_frame().T], ignore_index=True)
    
    return summary, results
