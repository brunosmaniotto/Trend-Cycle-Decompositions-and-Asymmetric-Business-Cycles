import numpy as np
import cvxpy as cp
from statsmodels.tsa.filters import hp_filter

def l2_hp_filter(y, lamb=1600):
    """
    Standard Hodrick-Prescott filter (L2 penalty on deviations, L2 on smoothness).
    
    Parameters
    ----------
    y : array-like
        Input time series.
    lamb : float
        Smoothing parameter (default 1600 for quarterly data).
        
    Returns
    -------
    trend : np.ndarray
        The estimated trend component.
    cycle : np.ndarray
        The estimated cyclical component (y - trend).
    """
    cycle, trend = hp_filter.hpfilter(y, lamb=lamb)
    return trend, cycle

def l1_hp_filter(y, lamb=1600, solver=cp.CLARABEL):
    """
    L1-HP filter (L1 penalty on deviations, L1 penalty on second differences).
    
    This filter targets the conditional median of the series and produces 
    piecewise linear trends, making it robust to outliers and structural breaks.
    
    Objective:
        min sum(|y_t - tau_t|) + lambda * sum(|Delta^2 tau_t|)
    
    Parameters
    ----------
    y : array-like
        Input time series.
    lamb : float
        Smoothing parameter. Note that optimal lambda values for L1 filters
        typically differ from L2 filters.
    solver : cvxpy.solver, optional
        The solver to use for the convex optimization problem.
        
    Returns
    -------
    trend : np.ndarray
        The estimated trend component.
    cycle : np.ndarray
        The estimated cyclical component (y - trend).
    """
    y = np.asarray(y).flatten()
    T = len(y)
    
    # Optimization variable (Trend)
    tau = cp.Variable(T)
    
    # L1 penalty on deviations from trend (Fit term)
    fit_cost = cp.norm1(y - tau)
    
    # L1 penalty on second differences (Smoothness term)
    # D2 is the second difference operator
    if T > 2:
        # Construct second differences efficiently
        # tau[t] - 2*tau[t-1] + tau[t-2]
        # In vector form: tau[2:] - 2*tau[1:-1] + tau[:-2]
        second_diff = tau[2:] - 2*tau[1:-1] + tau[:-2]
        smooth_cost = cp.norm1(second_diff)
    else:
        smooth_cost = 0
        
    # Objective function
    objective = cp.Minimize(fit_cost + lamb * smooth_cost)
    
    # Solve problem
    problem = cp.Problem(objective)
    try:
        problem.solve(solver=solver, verbose=False)
    except cp.SolverError:
        # Fallback to ECOS if CLARABEL fails or isn't available
        problem.solve(solver=cp.ECOS, verbose=False)
    
    if tau.value is None:
        raise RuntimeError("Optimization failed to converge.")
        
    trend = tau.value
    cycle = y - trend
    
    return trend, cycle
