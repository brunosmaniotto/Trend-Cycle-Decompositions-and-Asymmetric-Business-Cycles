import numpy as np
import cvxpy as cp

def huber_hp_filter(y, lamb=1600, delta=1.0, solver=cp.CLARABEL):
    y = np.asarray(y).flatten()
    T = len(y)
    tau = cp.Variable(T)
    
    # L1 fit
    fit_cost = cp.norm(y - tau, 1)
    
    if T > 2:
        second_diff = tau[2:] - 2*tau[1:-1] + tau[:-2]
        # Huber penalty
        # cp.huber(x, M) sum is convex
        smooth_cost = cp.sum(cp.huber(second_diff, M=delta))
    else:
        smooth_cost = 0
        
    objective = cp.Minimize(fit_cost + lamb * smooth_cost)
    problem = cp.Problem(objective)
    
    try:
        problem.solve(solver=solver, verbose=False)
    except Exception as e:
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception as e2:
            return np.full(T, np.nan), np.full(T, np.nan)
            
    if tau.value is None:
            return np.full(T, np.nan), np.full(T, np.nan)

    return tau.value, y - tau.value

def elastic_hp_filter(y, lamb=1600, alpha=0.5, solver=cp.CLARABEL):
    y = np.asarray(y).flatten()
    T = len(y)
    tau = cp.Variable(T)
    
    fit_cost = cp.norm(y - tau, 1)
    
    if T > 2:
        second_diff = tau[2:] - 2*tau[1:-1] + tau[:-2]
        # Elastic Net: alpha * L1 + (1-alpha) * L2^2
        # Note: cp.norm(x, 1) is convex. cp.sum_squares(x) is convex.
        # Sum of convex is convex.
        smooth_cost = alpha * cp.norm(second_diff, 1) + (1 - alpha) * cp.sum_squares(second_diff)
    else:
        smooth_cost = 0
        
    objective = cp.Minimize(fit_cost + lamb * smooth_cost)
    problem = cp.Problem(objective)
    
    try:
        problem.solve(solver=solver, verbose=False)
    except Exception as e:
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
        except Exception as e2:
            return np.full(T, np.nan), np.full(T, np.nan)
            
    if tau.value is None:
            return np.full(T, np.nan), np.full(T, np.nan)
        
    return tau.value, y - tau.value