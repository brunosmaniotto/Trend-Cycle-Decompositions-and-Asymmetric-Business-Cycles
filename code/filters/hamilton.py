import numpy as np
import cvxpy as cp

def l2_hamilton_filter(y, h=8, p=4):
    """
    Standard Hamilton (2018) filter using OLS regression.
    
    Model: y_{t+h} = alpha + sum_{j=0}^{p-1} beta_j * y_{t-j} + epsilon_{t+h}
    Cycle = epsilon_{t+h}
    
    Parameters
    ----------
    y : array-like
        Input time series.
    h : int
        Forecast horizon (default 8 for quarterly data).
    p : int
        Number of lags (default 4 for quarterly data).
        
    Returns
    -------
    trend : np.ndarray
        The estimated trend component (fitted values).
        Note: The first p+h-1 values will be NaN.
    cycle : np.ndarray
        The estimated cyclical component (residuals).
        Note: The first p+h-1 values will be NaN.
    """
    y = np.asarray(y).flatten()
    T = len(y)
    
    # Prepare X and Y matrices
    # Target: y_{t+h}
    # Features: constant, y_{t}, y_{t-1}, ..., y_{t-p+1}
    
    # Valid indices for regression start at t where (t-p+1) >= 0 -> t >= p-1
    # But we are predicting t+h, so the prediction starts at index p-1+h
    
    # Y_target starts at index p+h-1
    y_target = y[h+p-1:]
    n_obs = len(y_target)
    
    if n_obs <= 0:
        raise ValueError(f"Time series length {T} is too short for h={h}, p={p}")
    
    X = np.ones((n_obs, p + 1))
    for j in range(p):
        # Lag j corresponds to y_{t-j}.
        # If y_target[0] is y_{h+p-1}, then t = p-1.
        # We need y_{p-1-j}.
        # The sequence of features for lag j is y[p-1-j : T-h-j]
        X[:, j+1] = y[p-1-j : T-h-j]
        
    # OLS Solution: beta = (X'X)^-1 X'y
    # Using lstsq for stability
    beta, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)
    
    # Construct Trend and Cycle
    trend = np.full(T, np.nan)
    cycle = np.full(T, np.nan)
    
    # Fitted values
    y_fitted = X @ beta
    
    trend[h+p-1:] = y_fitted
    cycle[h+p-1:] = y_target - y_fitted
    
    return trend, cycle

def l1_hamilton_filter(y, h=8, p=4, solver=cp.CLARABEL):
    """
    L1-Hamilton filter using Least Absolute Deviations (LAD) regression.
    
    Minimizes sum(|y_{t+h} - alpha - sum beta_j * y_{t-j}|).
    
    Parameters
    ----------
    y : array-like
        Input time series.
    h : int
        Forecast horizon.
    p : int
        Number of lags.
    solver : cvxpy.solver, optional
        Solver for the convex problem.
        
    Returns
    -------
    trend : np.ndarray
        Trend component (NaN padded).
    cycle : np.ndarray
        Cyclical component (NaN padded).
    """
    y = np.asarray(y).flatten()
    T = len(y)
    
    y_target = y[h+p-1:]
    n_obs = len(y_target)
    
    if n_obs <= 0:
        raise ValueError(f"Time series length {T} is too short for h={h}, p={p}")
        
    X = np.ones((n_obs, p + 1))
    for j in range(p):
        X[:, j+1] = y[p-1-j : T-h-j]
        
    # Optimization variables
    beta = cp.Variable(p + 1)
    
    # L1 Loss
    loss = cp.norm1(y_target - X @ beta)
    
    problem = cp.Problem(cp.Minimize(loss))
    try:
        problem.solve(solver=solver, verbose=False)
    except cp.SolverError:
        problem.solve(solver=cp.ECOS, verbose=False)
        
    if beta.value is None:
        raise RuntimeError("Optimization failed to converge.")
        
    # Reconstruct
    trend = np.full(T, np.nan)
    cycle = np.full(T, np.nan)
    
    y_fitted = X @ beta.value
    trend[h+p-1:] = y_fitted
    cycle[h+p-1:] = y_target - y_fitted
    
    return trend, cycle
