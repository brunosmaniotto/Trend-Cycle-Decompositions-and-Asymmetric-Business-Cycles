import numpy as np
import pandas as pd
import os

# Add parent directory to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from filters.hp import l1_hp_filter, l2_hp_filter
from filters.hamilton import l1_hamilton_filter, l2_hamilton_filter
from models.structural import (
    PluckingModel, NetworkModel, HANKModel, GranularStructuralModel,
    FinancialAcceleratorModel, ZLBModel, AsymmetricRBCModel,
    UncertaintyShocksModel, HysteresisModel
)

def apply_filters_to_model(model_output):
    """
    Apply all filters to model output and compute performance metrics.
    """
    y = model_output['output']
    true_trend = model_output['potential']
    true_cycle = model_output['cycle']
    
    results = {}
    
    # Define filters
    filter_funcs = {
        'HP-L2': (l2_hp_filter, {'lamb': 1600}),
        'HP-L1': (l1_hp_filter, {'lamb': 600}), # Using a generally good lambda for L1-HP
        'Hamilton-L2': (l2_hamilton_filter, {}),
        'Hamilton-L1': (l1_hamilton_filter, {})
    }
    
    for name, (func, kwargs) in filter_funcs.items():
        try:
            trend_est, cycle_est = func(y, **kwargs)
            
            # Metrics
            valid = ~np.isnan(trend_est)
            if valid.sum() > 0:
                bias = np.mean(trend_est[valid] - true_trend[valid])
                rmse = np.sqrt(np.mean((trend_est[valid] - true_trend[valid])**2))
                
                valid_cycle = ~np.isnan(cycle_est)
                if valid_cycle.sum() > 3:
                    cycle_corr = np.corrcoef(cycle_est[valid_cycle], true_cycle[valid_cycle])[0,1]
                    cycle_skew = pd.Series(cycle_est[valid_cycle]).skew()
                else:
                    cycle_corr = np.nan
                    cycle_skew = np.nan
            else:
                bias = rmse = cycle_corr = cycle_skew = np.nan
                
            results[name] = {
                'bias': bias,
                'rmse': rmse,
                'cycle_corr': cycle_corr,
                'cycle_skew': cycle_skew,
            }
        except Exception as e:
            results[name] = {'bias': np.nan, 'rmse': np.nan, 'cycle_corr': np.nan, 'cycle_skew': np.nan}
            
    return results

def run_structural_experiments(T=200, n_reps=50):
    """
    Run experiments on all structural models.
    """
    models = {
        'Plucking': PluckingModel(),
        'Network': NetworkModel(n_sectors=50),
        'HANK': HANKModel(n_agents=1000),
        'Granular': GranularStructuralModel(n_firms=500),
        'FinAccel': FinancialAcceleratorModel(),
        'ZLB': ZLBModel(),
        'AsymRBC': AsymmetricRBCModel(),
        'UncertaintyShocks': UncertaintyShocksModel(),
        'Hysteresis': HysteresisModel()
    }
    
    aggregated_results = []
    
    for model_name, model in models.items():
        print(f"Simulating {model_name} Model ({n_reps} replications)...")
        for i in range(n_reps):
            np.random.seed(i)
            sim_data = model.simulate(T=T)
            res = apply_filters_to_model(sim_data)
            
            for filter_name, metrics in res.items():
                aggregated_results.append({
                    'Model': model_name,
                    'Filter': filter_name,
                    'Rep': i,
                    'Bias': metrics['bias'],
                    'RMSE': metrics['rmse'],
                    'Cycle Corr': metrics['cycle_corr'],
                    'Cycle Skew': metrics['cycle_skew']
                })
    
    df = pd.DataFrame(aggregated_results)
    
    # Save
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(script_dir, '../../output/results'))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving structural experiments results to: {os.path.join(output_dir, 'structural_experiments_results.csv')}")
    
    df.to_csv(os.path.join(output_dir, 'structural_experiments_results.csv'), index=False)
    
    # Summary
    summary = df.groupby(['Model', 'Filter'])[['Bias', 'RMSE', 'Cycle Skew']].mean().reset_index()
    print("\nStructural Model Results Summary:")
    print(summary)
    
    return summary
