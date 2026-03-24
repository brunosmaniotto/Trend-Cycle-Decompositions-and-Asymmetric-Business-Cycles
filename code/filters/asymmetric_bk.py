import numpy as np
import cvxpy as cp
from scipy import signal

def construct_split_basis(T, low=6, high=32, n_freqs=20):
    """
    Construct a dictionary of half-wave rectified sine and cosine waves.
    
    Parameters
    ----------
    T : int
        Sample size.
    low, high : int
        Period cutoffs (quarters).
    n_freqs : int
        Number of frequencies to sample within the band.
        
    Returns
    -------
    D : np.ndarray
        Design matrix of shape (T, 4 * n_freqs).
        Columns: sin+, sin-, cos+, cos- for each freq.
    """
    omega_low = 2 * np.pi / high
    omega_high = 2 * np.pi / low
    
    freqs = np.linspace(omega_low, omega_high, n_freqs)
    t = np.arange(T)
    
    basis_list = []
    
    for omega in freqs:
        sin_wave = np.sin(omega * t)
        cos_wave = np.cos(omega * t)
        
        # Split into positive and negative parts (both non-negative)
        # We want D*beta to form the cycle.
        # Traditionally sin = sin+ - sin-. 
        # Here we allow independent weights: b1*sin+ + b2*sin-
        # This allows the 'up' part of the wave to have different amplitude than 'down'.
        
        basis_list.append(np.maximum(sin_wave, 0)) # sin+
        basis_list.append(np.maximum(-sin_wave, 0)) # sin-
        basis_list.append(np.maximum(cos_wave, 0)) # cos+
        basis_list.append(np.maximum(-cos_wave, 0)) # cos-
        
    return np.column_stack(basis_list)

def asymmetric_bk_filter(y, low=6, high=32, n_freqs=20, method='L1'):
    """
    Asymmetric Baxter-King Filter via Split-Basis Regression.
    
    Models the cycle as a linear combination of rectified sine/cosine waves.
    y_t = trend_t + cycle_t
    cycle_t = D * beta
    
    We solve: min ||y - trend - D*beta|| + regularization
    
    Simplified approach: Assume trend is smooth (e.g. linear/quadratic) or removed.
    We regress DETRENDED y (or HP-trend-removed y) on the basis to see if it fits better.
    
    Alternatively, we can solve for trend and cycle simultaneously:
    min ||y - tau - D*beta||_1 + lambda ||D^2 tau||_1
    
    For this demo, we will assume we are filtering an already roughly detrended series
    (or fitting the cycle directly to y if y is stationary-ish) to demonstrate the *shape* matching.
    
    Let's assume we fit to the HP-L1 cycle to see if ABK can approximate it with waves.
    
    Parameters
    ----------
    y : array-like
        Time series.
    method : str
        'L1' (Robust) or 'L2' (OLS).
        
    Returns
    -------
    cycle : np.ndarray
        Estimated asymmetric cycle.
    """
    y = np.asarray(y).flatten()
    T = len(y)
    
    # Construct basis
    D = construct_split_basis(T, low, high, n_freqs)
    k = D.shape[1]
    
    # Optimization
    beta = cp.Variable(k)
    
    # To allow for a trend, let's include a polynomial trend in the regression
    # or just center y.
    # Let's add a quadratic trend to the basis to make it self-contained
    t = np.arange(T)
    Trend_Basis = np.column_stack([np.ones(T), t, t**2])
    
    # Full design matrix
    # Cycle = D @ beta
    # Trend = Trend_Basis @ gamma
    gamma = cp.Variable(3)
    
    residuals = y - D @ beta - Trend_Basis @ gamma
    
    if method == 'L1':
        # Sparse usage of basis is also good (Lasso-like on beta?)
        # Let's just minimize LAD of residuals
        obj = cp.norm(residuals, 1) + 0.1 * cp.norm(beta, 1) # Elastic net on weights to prevent overfitting
    else:
        obj = cp.sum_squares(residuals) + 0.1 * cp.sum_squares(beta)
        
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=cp.CLARABEL)
    
    cycle = (D @ beta).value
    trend = (Trend_Basis @ gamma).value
    
    return trend, cycle

def standard_bk_filter(y, low=6, high=32, K=12):
    """
    Standard Baxter-King filter (wrapper).
    Returns cycle only (trend is residue).
    """
    from statsmodels.tsa.filters.bk_filter import bkfilter
    # bkfilter returns a pandas series/dataframe
    cycle = bkfilter(y, low, high, K)
    # Pad with NaNs to match original length
    cycle_padded = np.full(len(y), np.nan)
    # Statsmodels BK truncates K from both ends
    if len(cycle) == len(y) - 2*K:
        cycle_padded[K:-K] = cycle
    elif len(cycle) == len(y):
        # statsmodels returned full-length output (already padded or no truncation)
        cycle_padded = np.asarray(cycle).flatten()
    elif len(cycle) < len(y):
        # Truncated output of unknown length — center it
        trim = len(y) - len(cycle)
        left = trim // 2
        cycle_padded[left:left + len(cycle)] = np.asarray(cycle).flatten()
    else:
        # Longer than input (shouldn't happen) — take the center
        excess = len(cycle) - len(y)
        start = excess // 2
        cycle_padded = np.asarray(cycle).flatten()[start:start + len(y)]
    return cycle_padded
