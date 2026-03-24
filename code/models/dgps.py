import numpy as np
from scipy import stats

class DGP:
    """Base class for Data Generating Processes"""
    def generate(self, T, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return self._generate(T)
    
    def _generate(self, T):
        raise NotImplementedError

def generate_garch_process(T, omega=0.1, alpha=0.1, beta=0.8):
    """Helper to generate GARCH(1,1) volatility"""
    sigma2 = np.zeros(T)
    eps = np.zeros(T)
    # Initialize with unconditional variance
    sigma2[0] = omega / (1 - alpha - beta)
    
    # Generate shocks
    z = np.random.randn(T)
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
        
    return eps, np.sqrt(sigma2)

def generate_skewed_garch_process(T, omega=0.1, alpha=0.1, beta=0.8, skew_a=-5):
    """Helper to generate GARCH(1,1) volatility with Skewed Shocks"""
    sigma2 = np.zeros(T)
    eps = np.zeros(T)
    # Initialize with unconditional variance
    sigma2[0] = omega / (1 - alpha - beta)
    
    # Generate skewed shocks (normalized to mean 0, var 1 approx for scaling)
    z_raw = stats.skewnorm.rvs(skew_a, size=T)
    z = (z_raw - z_raw.mean()) / z_raw.std()
    
    eps[0] = np.sqrt(sigma2[0]) * z[0]
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
        
    return eps, np.sqrt(sigma2)

def generate_skewed_fat_tail_shocks(T, df, skewness_param, loc=0, scale=1):
    """
    Helper to generate shocks that are both skewed and fat-tailed.
    Uses a transformation of t-distributed variables.
    """
    t_shocks = stats.t.rvs(df, size=T)
    
    # Apply skewness transformation: scale positive and negative parts differently
    # A simple, heuristic transformation to induce skewness
    # Positive skew: right tail heavier; Negative skew: left tail heavier
    # We use skewness_param > 0 for right skew, < 0 for left skew
    
    # Make positive values larger/smaller relative to negative values
    if skewness_param > 0: # Right skew
        skewed_t_shocks = np.where(t_shocks > 0, t_shocks * (1 + abs(skewness_param)), t_shocks)
    else: # Left skew (negative skewness_param)
        skewed_t_shocks = np.where(t_shocks < 0, t_shocks * (1 + abs(skewness_param)), t_shocks)
        
    # Normalize to mean 0 and variance 1 for consistent scaling with 'scale'
    skewed_t_shocks = (skewed_t_shocks - np.mean(skewed_t_shocks)) / np.std(skewed_t_shocks)
    
    return loc + scale * skewed_t_shocks

# --- A. The "Null" Tests ---

class RandomWalkDGP(DGP):
    """1. Pure Random Walk (Hamilton)"""
    def __init__(self, sigma=1):
        self.sigma = sigma
    
    def _generate(self, T):
        shocks = np.random.randn(T) * self.sigma
        y = np.cumsum(shocks)
        true_trend = y.copy() # In this context, the cycle is the random walk
        true_cycle = np.zeros(T) 
        return y, true_trend, true_cycle

class DriftRandomWalkDGP(DGP):
    """2. Random Walk + Drift (Hamilton)"""
    def __init__(self, drift=1, sigma=1):
        self.drift = drift
        self.sigma = sigma
        
    def _generate(self, T):
        shocks = np.random.randn(T) * self.sigma
        true_trend = np.cumsum(np.full(T, self.drift))
        y = true_trend + np.cumsum(shocks) # Data = drift_trend + RW_noise
        true_cycle = y - true_trend # The noise part
        return y, true_trend, true_cycle

# --- B. The "Ideal" Tests ---

class UCModelDGP(DGP):
    """3. Hodrick's Unobserved Components Model (Symmetric Baseline)"""
    def __init__(self, rho=0.7, sigma_eta=0.5, sigma_eps=0.5):
        self.rho = rho # Cycle persistence
        self.sigma_eta = sigma_eta # Trend shock
        self.sigma_eps = sigma_eps # Cycle shock
        
    def _generate(self, T):
        # Smooth Trend: I(2) process (random walk drift)
        eta = np.random.randn(T) * self.sigma_eta
        g = np.cumsum(eta)
        true_trend = np.cumsum(g)
        
        # Cycle: AR(1)
        eps = np.random.randn(T) * self.sigma_eps
        true_cycle = np.zeros(T)
        for t in range(1, T):
            true_cycle[t] = self.rho * true_cycle[t-1] + eps[t]
            
        y = true_trend + true_cycle
        return y, true_trend, true_cycle

# --- C. The "Robustness" Tests ---

class FatTailedRWDGP(DGP):
    """4. Fat-Tailed Random Walk (Outlier Robustness)"""
    def __init__(self, df=3, sigma=1):
        self.df = df
        self.sigma = sigma
        
    def _generate(self, T):
        # Student-t shocks
        shocks = stats.t.rvs(self.df, size=T) * self.sigma
        y = np.cumsum(shocks)
        true_trend = y.copy() # The cycle is the random walk
        true_cycle = np.zeros(T) # Correction: No cycle in RW
        return y, true_trend, true_cycle

class AsymmetricRWDGP(DGP):
    """4b. Asymmetric Random Walk (Skewed Shocks)"""
    def __init__(self, a=-5, sigma=1):
        self.a = a # Skewness parameter
        self.sigma = sigma
        
    def _generate(self, T):
        # Skew-normal shocks
        shocks = stats.skewnorm.rvs(self.a, loc=0, scale=self.sigma, size=T)
        y = np.cumsum(shocks)
        true_trend = y.copy()
        true_cycle = np.zeros(T)
        return y, true_trend, true_cycle

class SkewedUCDGP(DGP):
    """5. Skewed Cycle UC Model"""
    def __init__(self, rho=0.7, sigma_eta=0.5, skew_a=-5, sigma_eps=0.5):
        self.rho = rho
        self.sigma_eta = sigma_eta
        self.skew_a = skew_a
        self.sigma_eps = sigma_eps
        
    def _generate(self, T):
        # Trend (Symmetric I(2))
        eta = np.random.randn(T) * self.sigma_eta
        g = np.cumsum(eta)
        true_trend = np.cumsum(g)
        
        # Cycle: AR(1) with Skewed Shocks
        eps = stats.skewnorm.rvs(self.skew_a, size=T) * self.sigma_eps
        # Normalize eps to have mean 0 (important for true_cycle)
        eps = eps - eps.mean()
        
        true_cycle = np.zeros(T)
        for t in range(1, T):
            true_cycle[t] = self.rho * true_cycle[t-1] + eps[t]
            
        y = true_trend + true_cycle
        return y, true_trend, true_cycle

class SkewedFatTailDGP(DGP):
    """11. Skewed & Fat-Tailed Cycle"""
    def __init__(self, rho=0.7, sigma_eta=0.5, df=5, skew_param=-1.0, scale_eps=1.0):
        self.rho = rho
        self.sigma_eta = sigma_eta
        self.df = df
        self.skew_param = skew_param
        self.scale_eps = scale_eps
        
    def _generate(self, T):
        # Trend (Symmetric I(2))
        eta = np.random.randn(T) * self.sigma_eta
        g = np.cumsum(eta)
        true_trend = np.cumsum(g)
        
        # Cycle: AR(1) with Skewed and Fat-Tailed Shocks
        eps = generate_skewed_fat_tail_shocks(T, self.df, self.skew_param, scale=self.scale_eps)
        
        true_cycle = np.zeros(T)
        for t in range(1, T):
            true_cycle[t] = self.rho * true_cycle[t-1] + eps[t]
            
        y = true_trend + true_cycle
        return y, true_trend, true_cycle


# --- D. The "Gap Recoverability" & Asymmetric Structure ---

class PhaseShiftDGP(DGP):
    """6. Phase Shift Test (Sine Wave)"""
    def __init__(self, period=40, amplitude=5):
        self.period = period
        self.amplitude = amplitude
        
    def _generate(self, T):
        t = np.arange(T)
        true_trend = 0.5 * t # Linear trend
        true_cycle = self.amplitude * np.sin(2 * np.pi * t / self.period)
        y = true_trend + true_cycle + np.random.randn(T) * 0.1 # Add some noise
        return y, true_trend, true_cycle

class StatisticalPluckingDGP(DGP):
    """7. Statistical Plucking (Censored Cycle)"""
    def __init__(self, rho=0.8, sigma_cycle=1.0, trend_growth=0.5):
        self.rho = rho
        self.sigma_cycle = sigma_cycle
        self.trend_growth = trend_growth
        
    def _generate(self, T):
        # Trend: Linear growth
        true_trend = np.cumsum(np.full(T, self.trend_growth)) + 100
        
        # Cycle: AR(1) but strictly negative (Censored)
        raw_cycle = np.zeros(T)
        eps = np.random.randn(T) * self.sigma_cycle
        for t in range(1, T):
            raw_cycle[t] = self.rho * raw_cycle[t-1] + eps[t]
            
        # Plucking logic: Cycle must be <= 0. Potential is the top of the envelope.
        true_cycle = np.minimum(0, raw_cycle)
        
        y = true_trend + true_cycle + np.random.randn(T) * 0.1 # Add some noise
        return y, true_trend, true_cycle

class SharpCrashDGP(DGP):
    """8. Sharp Crash / Slow Recovery (Sawtooth)"""
    def __init__(self, period=50, amplitude=5):
        self.period = period
        self.amplitude = amplitude
        
    def _generate(self, T):
        t = np.arange(T)
        true_trend = 0.5 * t
        
        # Sawtooth wave: Rises linearly, drops instantly
        from scipy import signal
        # sawtooth(t, width=1) gives /|/| (slow rise, sharp drop)
        cycle_raw = signal.sawtooth(2 * np.pi * t / self.period, width=1.0)
        
        # Scale and shift to be mean zero approx and negative skew
        true_cycle = (cycle_raw - np.mean(cycle_raw)) * self.amplitude
        
        y = true_trend + true_cycle + np.random.randn(T) * 0.1 # Add some noise
        return y, true_trend, true_cycle

# --- E. Volatility / Heteroskedasticity ---

class GarchRWDGP(DGP):
    """9. GARCH(1,1) Random Walk (Volatility Clustering)"""
    def __init__(self, omega=0.1, alpha=0.1, beta=0.8):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        
    def _generate(self, T):
        shocks, _ = generate_garch_process(T, self.omega, self.alpha, self.beta)
        y = np.cumsum(shocks)
        true_trend = y.copy()
        true_cycle = np.zeros(T)
        return y, true_trend, true_cycle

class AsymmetricGarchDGP(DGP):
    """10. Asymmetric GARCH Cycle (Skewness + Volatility Clustering)"""
    def __init__(self, omega=0.1, alpha=0.1, beta=0.8, skew_a=-5):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.skew_a = skew_a
        
    def _generate(self, T):
        # Smooth Trend
        t = np.arange(T)
        true_trend = 0.5 * t
        
        # Cycle: GARCH volatility * Skewed shocks
        cycle_shocks, _ = generate_skewed_garch_process(T, self.omega, self.alpha, self.beta, self.skew_a)
        true_cycle = cycle_shocks
        
        y = true_trend + true_cycle
        return y, true_trend, true_cycle
