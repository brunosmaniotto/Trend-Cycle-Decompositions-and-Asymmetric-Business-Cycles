import numpy as np
from scipy import stats
import pandas as pd

class AsymmetryTests:
    """
    Collection of statistical tests for business cycle asymmetry
    """
    
    @staticmethod
    def skewness_test(x, bootstrap_samples=10000):
        """
        Test if skewness is significantly different from zero
        H0: Skewness = 0 (symmetric)
        H1: Skewness != 0 (asymmetric)
        """
        n = len(x)
        skew = stats.skew(x)
        
        # Analytical standard error (assumes normality)
        if n > 2:
            se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
            z_stat = skew / se_skew
            p_value_analytical = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            return {'skewness': np.nan}
        
        # Bootstrap for robustness
        if bootstrap_samples > 0:
            bootstrap_skews = []
            for _ in range(bootstrap_samples):
                sample = np.random.choice(x, size=n, replace=True)
                bootstrap_skews.append(stats.skew(sample))
            
            bootstrap_skews = np.array(bootstrap_skews)
            # One-sided p-value * 2 for two-sided
            p_value_bootstrap = np.mean(np.abs(bootstrap_skews) >= np.abs(skew))
            ci = np.percentile(bootstrap_skews, [2.5, 97.5])
        else:
            p_value_bootstrap = np.nan
            ci = [np.nan, np.nan]
        
        return {
            'skewness': skew,
            'z_statistic': z_stat,
            'p_value_analytical': p_value_analytical,
            'p_value_bootstrap': p_value_bootstrap,
            'reject_h0_5pct': p_value_analytical < 0.05,
            'bootstrap_ci': ci
        }
    
    @staticmethod
    def triples_test(x):
        """
        Randles et al. (1980) Triples Test
        Tests for symmetry using data triples. More robust than skewness.
        """
        n = len(x)
        if n > 500: # Sampling for performance on large arrays
             x = np.random.choice(x, 500, replace=False)
             n = 500
             
        triples_sum = 0
        count = 0
        
        # Calculate test statistic from all triples
        # This is O(N^3), so be careful with large N
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    xi, xj, xk = x[i], x[j], x[k]
                    median_triple = np.median([xi, xj, xk])
                    mean_triple = np.mean([xi, xj, xk])
                    triples_sum += np.sign(median_triple - mean_triple)
                    count += 1
        
        eta = triples_sum / count
        var_eta = 1 / count  # Under null hypothesis
        z_stat = eta / np.sqrt(var_eta)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'eta': eta,
            'z_statistic': z_stat,
            'p_value': p_value,
            'reject_h0_5pct': p_value < 0.05
        }
    
    @staticmethod
    def sichel_test(x):
        """
        Sichel (1993) test for deepness and steepness
        Deepness: Asymmetry in levels (troughs deeper than peaks)
        Steepness: Asymmetry in first differences (sharp declines, gradual recoveries)
        """
        n = len(x)
        
        # Deepness test (asymmetry in levels)
        x_centered = x - np.mean(x)
        denom = np.mean(x_centered**2)**1.5
        deepness = np.mean(x_centered**3) / denom if denom > 0 else 0
        se_deepness = np.sqrt(6 / n)  # Under normality
        z_deepness = deepness / se_deepness
        p_deepness = 2 * (1 - stats.norm.cdf(abs(z_deepness)))
        
        # Steepness test (asymmetry in differences)
        dx = np.diff(x)
        dx_centered = dx - np.mean(dx)
        denom_dx = np.mean(dx_centered**2)**1.5
        steepness = np.mean(dx_centered**3) / denom_dx if denom_dx > 0 else 0
        se_steepness = np.sqrt(6 / (n-1))
        z_steepness = steepness / se_steepness
        p_steepness = 2 * (1 - stats.norm.cdf(abs(z_steepness)))
        
        return {
            'deepness': deepness,
            'deepness_z': z_deepness,
            'deepness_p': p_deepness,
            'steepness': steepness,
            'steepness_z': z_steepness,
            'steepness_p': p_steepness,
            'reject_deepness_5pct': p_deepness < 0.05,
            'reject_steepness_5pct': p_steepness < 0.05
        }

def rolling_asymmetry_test(data, window=40, step=4):
    """
    Calculate rolling window asymmetry statistics
    """
    results = {'date_idx': [], 'skewness': [], 'p_value': [], 'deepness': [], 'steepness': []}
    tester = AsymmetryTests()
    
    for i in range(0, len(data) - window, step):
        window_data = data[i:i+window]
        
        # Detrend window (Linear trend removal for local cycle)
        t = np.arange(len(window_data))
        z = np.polyfit(t, window_data, 1)
        p = np.poly1d(z)
        cycle = window_data - p(t)
        
        # Run tests
        skew_test = tester.skewness_test(cycle, bootstrap_samples=0) # Analytical for speed
        sichel = tester.sichel_test(cycle)
        
        results['date_idx'].append(i + window//2)
        results['skewness'].append(skew_test['skewness'])
        results['p_value'].append(skew_test['p_value_analytical'])
        results['deepness'].append(sichel['deepness'])
        results['steepness'].append(sichel['steepness'])
    
    return pd.DataFrame(results)
