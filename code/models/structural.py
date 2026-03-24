import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import t as student_t, skewnorm

class StructuralModel:
    """Base class for Structural Models"""
    def simulate(self, T=200, burn_in=50):
        raise NotImplementedError

class PluckingModel(StructuralModel):
    """
    Dupraz, Nakamura, Steinsson (2020) Plucking Model
    - Downward nominal wage rigidity creates asymmetric cycles
    """
    def __init__(self, params=None):
        if params is None:
            params = {
                'beta': 0.99, 'sigma': 2.0, 'phi': 2.0, 'theta': 6.0,
                'rho_a': 0.9, 'sigma_a': 0.01, 'rho_d': 0.8, 'sigma_d': 0.02,
                'gamma': 0.0, 'pi_bar': 0.02/4, 'g': 0.005
            }
        self.params = params
        
    def simulate(self, T=200, burn_in=50):
        p = self.params
        total_T = T + burn_in
        y, l, w, a, d, y_potential = np.zeros(total_T), np.zeros(total_T), np.zeros(total_T), np.zeros(total_T), np.zeros(total_T), np.zeros(total_T)
        l_ss = ((p['theta']-1)/p['theta'])**(1/(p['phi']+p['sigma']))
        y_ss = l_ss
        w_ss = 1.0
        y[0], l[0], w[0] = y_ss, l_ss, w_ss
        
        for t in range(1, total_T):
            a[t] = p['rho_a'] * a[t-1] + np.random.randn() * p['sigma_a']
            d[t] = p['rho_d'] * d[t-1] + np.random.randn() * p['sigma_d']
            y_potential[t] = y_potential[t-1] + p['g'] + a[t]
            w_flex = w[t-1] * (1 + p['pi_bar']) * np.exp((p['sigma'] + p['phi']) * d[t])
            w_lower = w[t-1] * (1 + p['gamma'])
            if w_flex < w_lower:
                w[t] = w_lower
                labor_gap = -p['sigma_d'] * (w[t] - w_flex) / w_flex
                y[t] = y_potential[t] + labor_gap
            else:
                w[t] = w_flex
                y[t] = y_potential[t]
        
        y = y[burn_in:]
        y_potential = y_potential[burn_in:]
        cycle = y - y_potential
        trend = np.cumsum(np.full(T, p['g'])) + 100
        y_level = trend + cycle
        return {'output': y_level, 'potential': trend, 'cycle': cycle}

class NetworkModel(StructuralModel):
    """
    Baqaee & Farhi (2020) Network Model
    """
    def __init__(self, n_sectors=50, params=None):
        self.n = n_sectors
        if params is None:
            params = {
                'sigma': 0.5, 'alpha': 0.5, 'network_density': 0.2,
                'rho_z': 0.9, 'sigma_z': 0.02, 'g': 0.005
            }
        self.params = params
        self.W = self._generate_network()
        self.leontief = linalg.inv(np.eye(self.n) - self.W)
        self.sector_sizes = np.random.pareto(2, self.n)
        self.sector_sizes = self.sector_sizes / self.sector_sizes.sum()
        
    def _generate_network(self):
        n, density = self.n, self.params['network_density']
        W = np.random.rand(n, n)
        W = (W < density).astype(float) * np.random.uniform(0.1, 0.4, (n, n))
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return W / row_sums * 0.5
    
    def simulate(self, T=200, burn_in=50):
        p = self.params
        total_T = T + burn_in
        z = np.zeros((total_T, self.n))
        y_agg = np.zeros(total_T)
        
        for t in range(1, total_T):
            z[t] = p['rho_z'] * z[t-1] + np.random.randn(self.n) * p['sigma_z']
            if p['sigma'] < 1:
                shock_effect = np.zeros(self.n)
                for i in range(self.n):
                    if z[t, i] < 0:
                        shock_effect += self.leontief[:, i] * z[t, i] * (1 + 0.5 * (1 - p['sigma']))
                    else:
                        shock_effect += self.leontief[:, i] * z[t, i] * p['sigma']
                y_sectors = np.exp(shock_effect)
            else:
                y_sectors = np.exp(self.leontief @ z[t])
            y_agg[t] = np.sum(self.sector_sizes * y_sectors)
        
        trend = np.cumsum(np.full(total_T, p['g']))
        y_agg = np.maximum(y_agg, 1e-10)
        y_level = np.log(y_agg) + trend + 100
        y_level = y_level[burn_in:]
        trend = trend[burn_in:] + 100
        cycle = y_level - trend
        return {'output': y_level, 'potential': trend, 'cycle': cycle}

class HANKModel(StructuralModel):
    """
    Simplified Heterogeneous Agent New Keynesian (HANK) Model.

    This is a reduced-form approximation inspired by Kaplan, Moll, and Violante
    (2018). Heterogeneity comes from a distribution of wealth levels that
    determines each agent's marginal propensity to consume (MPC).

    **Definition of potential output:** We define potential as the output path
    of a Representative Agent NK (RANK) economy where all agents have the
    unconstrained MPC (mpc_unconstrained = 0.1). This differs from the standard
    Kaplan-Moll-Violante flexible-price definition (which removes nominal
    rigidities but preserves heterogeneity). Our choice isolates the
    *amplification* effect of heterogeneous MPCs: the cycle measures the extra
    output volatility caused by hand-to-mouth agents relative to a
    permanent-income benchmark. This is a reduced-form approximation — a fully
    structural HANK potential would require solving for the flexible-price
    equilibrium with the full wealth distribution, which is beyond the scope
    of this reduced-form exercise.

    Asymmetry mechanism: During recessions (agg_shock < 0), the borrowing
    constraint binds for more agents (threshold shifts from borrowing_limit + 2
    to borrowing_limit + 3), raising the aggregate MPC and amplifying the
    downturn relative to the RANK counterfactual.
    """
    def __init__(self, n_agents=1000, params=None):
        self.n_agents = n_agents
        if params is None:
            params = {
                'borrowing_limit': -1.0, 'rho_agg': 0.8, 'sigma_agg': 0.02,
                'mpc_unconstrained': 0.1, 'mpc_constrained': 0.8, 'g': 0.005
            }
        self.params = params
        self.wealth = np.exp(np.random.randn(n_agents) * 2)
        
    def simulate(self, T=200, burn_in=50):
        p = self.params
        total_T = T + burn_in
        y_agg = np.zeros(total_T)
        y_potential_agg = np.zeros(total_T)
        agg_shock = np.zeros(total_T)
        wealth = self.wealth.copy()
        
        for t in range(1, total_T):
            agg_shock[t] = p['rho_agg'] * agg_shock[t-1] + np.random.randn() * p['sigma_agg']
            
            # HANK (Constrained) Consumption
            consumption = np.zeros(self.n_agents)
            income = np.exp(np.random.randn(self.n_agents) * 0.1 + agg_shock[t])
            
            # Identify constrained agents
            constrained = wealth < p['borrowing_limit'] + 2
            if agg_shock[t] < 0: # Recession tightens constraints
                 extra_constrained = wealth < p['borrowing_limit'] + 3
                 consumption[extra_constrained] = income[extra_constrained] * p['mpc_constrained']
                 consumption[~extra_constrained] = income[~extra_constrained] * p['mpc_unconstrained']
            else:
                 consumption[constrained] = income[constrained] * p['mpc_constrained']
                 consumption[~constrained] = income[~constrained] * p['mpc_unconstrained']
            
            y_agg[t] = consumption.mean()
            wealth = wealth + income - consumption
            wealth = np.maximum(wealth, p['borrowing_limit'])
            
            # RANK (Unconstrained) Consumption / Potential
            # Permanent income approx: C = Y (assuming standard consumption smoothing)
            # In a simple RANK with no frictions, C follows the shock process smoothed
            # Here we approximate Potential as the path if MPC was low/uniform (0.1) everywhere
            consumption_pot = income * p['mpc_unconstrained'] 
            y_potential_agg[t] = consumption_pot.mean()

        trend = np.cumsum(np.full(total_T, p['g'])) + 100
        y_level = np.log(np.maximum(y_agg, 1e-10)) + trend 
        y_pot_level = np.log(np.maximum(y_potential_agg, 1e-10)) + trend
        
        y_level = y_level[burn_in:]
        y_pot_level = y_pot_level[burn_in:]
        cycle = y_level - y_pot_level # Cycle is deviation from frictionless potential
        
        return {'output': y_level, 'potential': y_pot_level, 'cycle': cycle}

class GranularStructuralModel(StructuralModel):
    """Gabaix (2011) Granular Model"""
    def __init__(self, n_firms=500, params=None):
        self.n_firms = n_firms
        if params is None:
            params = {'alpha': 1.06, 'sigma_firm': 0.12, 'rho_firm': 0.8, 'entry_rate': 0.1, 'exit_threshold': -2.0, 'g': 0.005}
        self.params = params
        self.firm_sizes = self._initialize_firms()
    
    def _initialize_firms(self):
        ranks = np.arange(1, self.n_firms + 1)
        sizes = 1 / (ranks ** self.params['alpha'])
        return sizes / sizes.sum()
    
    def simulate(self, T=200, burn_in=50):
        p = self.params
        total_T = T + burn_in
        granular_residual = np.zeros(total_T)
        firm_sizes = self.firm_sizes.copy()
        firm_productivity = np.zeros(self.n_firms)
        
        for t in range(1, total_T):
            current_n = len(firm_sizes)
            firm_shocks = student_t.rvs(df=3, size=current_n) * p['sigma_firm']
            firm_productivity = p['rho_firm'] * firm_productivity + firm_shocks
            exits = firm_productivity < p['exit_threshold']
            
            exit_impact = 0
            if exits.sum() > 0:
                exit_impact = -np.sum(firm_sizes[exits] * 10)
                firm_sizes = firm_sizes[~exits]
                firm_productivity = firm_productivity[~exits]
                n_entrants = int(p['entry_rate'] * len(firm_sizes))
                if n_entrants > 0:
                    new_sizes = np.random.pareto(p['alpha'], n_entrants) * 0.001
                    total = firm_sizes.sum() + new_sizes.sum()
                    if total > 0: new_sizes /= total
                    firm_sizes = np.concatenate([firm_sizes*(1-new_sizes.sum()), new_sizes])
                    firm_productivity = np.concatenate([firm_productivity, np.zeros(n_entrants)])
            
            if len(firm_sizes) > 0:
                weights = firm_sizes ** 0.5
                weights /= weights.sum()
                granular_residual[t] = np.sum(weights * firm_productivity) + exit_impact
            
            # Asymmetry arises endogenously from power-law firm sizes,
            # exit dynamics, and fat-tailed (t(3)) shocks — no ad-hoc multiplier needed.
            
            if len(firm_sizes) > self.n_firms * 1.2:
                keep = np.argsort(firm_sizes)[-self.n_firms:]
                firm_sizes, firm_productivity = firm_sizes[keep], firm_productivity[keep]
                firm_sizes /= firm_sizes.sum()
                
        trend = np.cumsum(np.full(total_T, p['g'])) + 100
        y_agg = trend + granular_residual * 10
        y_agg = y_agg[burn_in:]
        trend = trend[burn_in:]
        return {'output': y_agg, 'potential': trend, 'cycle': y_agg - trend}

# --- NEW MODELS ---

class FinancialAcceleratorModel(StructuralModel):
    """
    Bernanke, Gertler, Gilchrist (1999) style Financial Accelerator
    Reduced form: Gap = rho*Gap + Shock + Financial_Wedge
    Wedge is asymmetric: rises sharply when Gap < 0
    """
    def __init__(self, params=None):
        if params is None:
            params = {
                'rho': 0.8, 'sigma': 1.0, 
                'phi': 0.1, # Reduced sensitivity to ensure stability (0.8 + 0.1 < 1.0)
                'g': 0.005
            }
        self.params = params
        
    def simulate(self, T=200, burn_in=50):
        p = self.params
        total_T = T + burn_in
        gap = np.zeros(total_T)
        potential = np.zeros(total_T)
        
        for t in range(1, total_T):
            shock = np.random.randn() * p['sigma']
            
            # Financial wedge: only active during downturns
            wedge = 0
            if gap[t-1] < 0:
                # Higher leverage cost when output is low
                wedge = -p['phi'] * abs(gap[t-1])
            
            # Output gap dynamics
            gap[t] = p['rho'] * gap[t-1] + shock + wedge
            
            potential[t] = potential[t-1] + p['g']
            
        trend = potential[burn_in:] + 100
        cycle = gap[burn_in:]
        output = trend + cycle
        
        return {'output': output, 'potential': trend, 'cycle': cycle}

class ZLBModel(StructuralModel):
    """
    New Keynesian Model with Zero Lower Bound
    """
    def __init__(self, params=None):
        if params is None:
            params = {
                'sigma': 1.0, 'kappa': 0.1, 'beta': 0.99,
                'phi_pi': 1.5, 'phi_y': 0.125,
                'r_natural': 0.01, # Natural rate
                'sigma_d': 0.02, 'rho_d': 0.8, # Demand shock
                'g': 0.005
            }
        self.params = params
        
    def simulate(self, T=200, burn_in=50):
        p = self.params
        total_T = T + burn_in
        
        gap = np.zeros(total_T)
        inflation = np.zeros(total_T)
        rate = np.zeros(total_T)
        demand_shock = np.zeros(total_T)
        
        potential = np.cumsum(np.full(total_T, p['g']))
        
        for t in range(1, total_T):
            demand_shock[t] = p['rho_d'] * demand_shock[t-1] + np.random.randn() * p['sigma_d']
            
            # Taylor rule (unconstrained)
            target_rate = p['r_natural'] + p['phi_pi'] * inflation[t-1] + p['phi_y'] * gap[t-1]
            
            # ZLB constraint
            rate[t] = max(0, target_rate)
            
            # IS Curve (backward approx)
            # If rate > target, real rate is too high -> contraction
            monetary_drag = (rate[t] - target_rate) # Positive if ZLB binds (rate too high)
            
            # Gap evolution: Reduced sensitivity (0.5) to prevent explosion
            gap[t] = 0.8 * gap[t-1] + demand_shock[t] - 0.5 * monetary_drag
            
            # Phillips curve
            inflation[t] = 0.5 * inflation[t-1] + p['kappa'] * gap[t]
            
        trend = potential[burn_in:] + 100
        cycle = gap[burn_in:]
        output = trend + cycle
        
        return {'output': output, 'potential': trend, 'cycle': cycle}

class AsymmetricRBCModel(StructuralModel):
    """
    Standard RBC with Asymmetric Shocks

    Uses skew-normal distribution for TFP shocks. Negative skewness (skew < 0)
    generates sharp contractions and gradual expansions. Each draw is centered
    by subtracting the theoretical mean of the skew-normal distribution so that
    E[eps] = 0 regardless of the skewness parameter. Sigma is scaled to ~0.02
    to match the shock magnitude of other structural models.
    """
    def __init__(self, params=None):
        if params is None:
            params = {'rho': 0.9, 'sigma': 0.02, 'skew': -5, 'g': 0.005}
        self.params = params
        # Pre-compute theoretical mean of skewnorm(a) for centering
        # E[X] = delta * sqrt(2/pi) where delta = a / sqrt(1 + a^2)
        a = params['skew']
        delta = a / np.sqrt(1 + a**2)
        self._skewnorm_mean = delta * np.sqrt(2 / np.pi)

    def simulate(self, T=200, burn_in=50):
        p = self.params
        total_T = T + burn_in
        cycle = np.zeros(total_T)
        potential = np.zeros(total_T)

        for t in range(1, total_T):
            # Skewed shock: draw scalar, subtract theoretical mean to center
            eps = (skewnorm.rvs(p['skew']) - self._skewnorm_mean) * p['sigma']

            cycle[t] = p['rho'] * cycle[t-1] + eps
            potential[t] = potential[t-1] + p['g']

        trend = potential[burn_in:] + 100
        cycle = cycle[burn_in:]
        output = trend + cycle

        return {'output': output, 'potential': trend, 'cycle': cycle}


class UncertaintyShocksModel(StructuralModel):
    """
    Bloom (2009) Uncertainty Shocks Model with Stochastic Volatility.

    Uncertainty shocks cause immediate contraction via a real-options freeze
    (firms delay hiring/investment when uncertainty is high) followed by
    gradual recovery as uncertainty mean-reverts. Asymmetry arises because
    uncertainty spikes are right-skewed: uncertainty jumps up sharply but
    declines slowly.

    Key equation:
        sigma_t = sigma_base * exp(v_t)
        v_t = rho_sigma * v_{t-1} + |eta_t|       (right-skewed innovations)
        y_t = rho * y_{t-1} + sigma_t * eps_t - phi * max(v_t, 0)
    The -phi * max(v_t, 0) term captures the real-options drag: high uncertainty
    directly reduces output even if the shock realization is zero.
    """
    def __init__(self, params=None):
        if params is None:
            params = {
                'rho_output': 0.9,     # Output persistence
                'sigma_base': 0.01,    # Baseline shock std
                'rho_sigma': 0.7,      # Uncertainty persistence
                'sigma_sigma': 0.1,    # Volatility of volatility
                'phi': 0.3,            # Real options drag coefficient
                'g': 0.005,            # Trend growth
            }
        self.params = params

    def simulate(self, T=200, burn_in=50):
        p = self.params
        total_T = T + burn_in
        gap = np.zeros(total_T)
        v = np.zeros(total_T)       # log-volatility process
        potential = np.zeros(total_T)

        for t in range(1, total_T):
            # Stochastic volatility: right-skewed innovations (abs value)
            eta = np.abs(np.random.randn()) * p['sigma_sigma']
            v[t] = p['rho_sigma'] * v[t-1] + eta

            # Time-varying volatility
            sigma_t = p['sigma_base'] * np.exp(v[t])

            # Output shock
            eps = np.random.randn() * sigma_t

            # Real options drag: high uncertainty depresses output
            drag = p['phi'] * max(v[t], 0)

            gap[t] = p['rho_output'] * gap[t-1] + eps - drag
            potential[t] = potential[t-1] + p['g']

        trend = potential[burn_in:] + 100
        cycle = gap[burn_in:]
        output = trend + cycle

        return {'output': output, 'potential': trend, 'cycle': cycle}


class HysteresisModel(StructuralModel):
    """
    Blanchard and Summers (1986) Hysteresis Model.

    Deep recessions permanently reduce potential output. This directly
    challenges the L1 "ceiling" assumption: if the ceiling falls during
    recessions, the L1 trend (piecewise-linear with kinks at recessions)
    may be biased upward because it doesn't track the downward shift in
    potential.

    Key equation:
        y*_t = y*_{t-1} + g + delta * min(c_{t-1}, 0)
    where delta controls hysteresis strength (fraction of negative gap
    that becomes permanent). When delta = 0, the model reduces to a
    standard constant-growth trend.

    Asymmetry mechanism: trend growth slows during recessions (via hysteresis)
    but does not accelerate symmetrically during expansions, producing
    negatively skewed output relative to a constant-growth trend.
    """
    def __init__(self, params=None):
        if params is None:
            params = {
                'rho': 0.8,        # Cycle persistence
                'sigma': 0.02,     # Demand shock std
                'delta': 0.2,      # Hysteresis strength (0 = none, 1 = full)
                'g': 0.005,        # Baseline trend growth
            }
        self.params = params

    def simulate(self, T=200, burn_in=50):
        p = self.params
        total_T = T + burn_in
        gap = np.zeros(total_T)
        potential = np.zeros(total_T)

        for t in range(1, total_T):
            # Demand shock
            eps = np.random.randn() * p['sigma']
            gap[t] = p['rho'] * gap[t-1] + eps

            # Hysteresis: negative gaps permanently reduce potential
            hysteresis_drag = p['delta'] * min(gap[t-1], 0)
            potential[t] = potential[t-1] + p['g'] + hysteresis_drag

        trend = potential[burn_in:] + 100
        cycle = gap[burn_in:]
        output = trend + cycle

        return {'output': output, 'potential': trend, 'cycle': cycle}
