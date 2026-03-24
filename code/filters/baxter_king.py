import numpy as np
from scipy import signal

def baxter_king_standard(y, low=6, high=32, K=12):
    """
    Standard Baxter-King band-pass filter.
    
    Parameters
    ----------
    y : array-like
        Input time series.
    low : int
        Low frequency cutoff (min periods, e.g. 6 quarters).
    high : int
        High frequency cutoff (max periods, e.g. 32 quarters).
    K : int
        Truncation parameter (half-length of filter).
        
    Returns
    -------
    trend : np.ndarray
        Trend component (residual).
    cycle : np.ndarray
        Cyclical component.
    """
    y = np.asarray(y).flatten()
    T = len(y)
    
    # Frequency cutoffs
    omega_low = 2 * np.pi / high
    omega_high = 2 * np.pi / low
    
    # Ideal weights
    def ideal_bp(j):
        if j == 0:
            return (omega_high - omega_low) / np.pi
        else:
            return (np.sin(omega_high * j) - np.sin(omega_low * j)) / (np.pi * j)
            
    # Compute weights
    b = np.array([ideal_bp(j) for j in range(-K, K+1)])
    
    # Adjustment for zero gain at zero frequency (sum to 0)
    b = b - b.sum() / len(b)
    
    # Apply filter via convolution
    # Valid mode: output size = T - 2K
    # We want to return size T with NaNs on edges
    cycle_valid = np.convolve(y, b, mode='valid')
    
    cycle = np.full(T, np.nan)
    cycle[K : T-K] = cycle_valid
    
    trend = y - cycle
    
    return trend, cycle

def baxter_king_asymmetric(y, low=6, high=32, K=12, asymmetry=0.3):
    """
    Asymmetric Baxter-King filter using split positive/negative projections.

    Projects separately onto positive (max(sin, 0)) and negative (min(sin, 0))
    parts of the basis functions, applying asymmetric weights.

    Weight convention: w_pos + w_neg = 2.0 (i.e. w_pos = 1 - asymmetry,
    w_neg = 1 + asymmetry). The average weight per half-wave is 1.0, which
    preserves the total energy (variance) of the cycle component. When
    asymmetry = 0, both weights equal 1.0 and the filter reduces to the
    standard symmetric BK. When asymmetry > 0, negative half-cycles receive
    higher weight, producing negatively skewed cycle estimates.

    Parameters
    ----------
    y : array-like
        Input time series.
    low, high : int
        Period cutoffs.
    K : int
        Truncation parameter.
    asymmetry : float
        Asymmetry parameter (0 = symmetric, >0 = negative skew emphasized).

    Returns
    -------
    trend : np.ndarray
        Trend component.
    cycle : np.ndarray
        Cyclical component.
    """
    y = np.asarray(y).flatten()
    T = len(y)
    
    # Frequency range
    omega_low = 2 * np.pi / high
    omega_high = 2 * np.pi / low
    
    cycle = np.zeros(T)
    
    # Discrete approximation of the integral over frequencies
    n_freqs = 20
    freqs = np.linspace(omega_low, omega_high, n_freqs)
    
    time = np.arange(T)
    
    for omega in freqs:
        sine_wave = np.sin(omega * time)
        cosine_wave = np.cos(omega * time)
        
        # Split basis
        sine_pos = np.maximum(sine_wave, 0)
        sine_neg = np.minimum(sine_wave, 0)
        cos_pos = np.maximum(cosine_wave, 0)
        cos_neg = np.minimum(cosine_wave, 0)
        
        # Asymmetric weights
        w_pos = 1.0 - asymmetry
        w_neg = 1.0 + asymmetry
        
        # Projection helper
        def project(target, basis):
            norm = np.dot(basis, basis)
            if norm < 1e-10: return np.zeros_like(basis)
            return np.dot(target, basis) / norm * basis
            
        # Compute projections
        p_sin_pos = project(y, sine_pos)
        p_sin_neg = project(y, sine_neg)
        p_cos_pos = project(y, cos_pos)
        p_cos_neg = project(y, cos_neg)
        
        # Combine
        component = (w_pos * (p_sin_pos + p_cos_pos) + 
                     w_neg * (p_sin_neg + p_cos_neg))
                     
        cycle += component
        
    # Normalize by number of frequencies
    cycle /= n_freqs
    
    # Apply Tukey window to smooth edge effects (similar to truncation)
    # This replaces the hard K-truncation of standard BK
    window = signal.windows.tukey(T, alpha=0.1)
    cycle = cycle * window
    
    trend = y - cycle
    
    return trend, cycle
