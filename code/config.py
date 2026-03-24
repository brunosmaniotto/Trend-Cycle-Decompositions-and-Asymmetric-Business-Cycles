"""
Shared configuration constants for the empirical analysis pipeline.

All empirical modules import filter parameters from here to ensure consistency.
"""

# HP filter smoothing parameters
LAMBDA_L2_STANDARD = 1600   # Standard HP-L2 (Ravn-Uhlig quarterly rule)
LAMBDA_L1_EMPIRICAL = 600   # L1-HP for empirical analysis (cross-validated)

# Hamilton filter defaults
HAMILTON_H = 8   # Forecast horizon (quarters)
HAMILTON_P = 4   # Number of lags

# Newey-West HAC standard errors
HAC_MAXLAGS = 4  # Default lag truncation for Newey-West

# FRED series dictionary
FRED_SERIES = {
    'GDPC1': 'Real GDP',
    'UNRATE': 'Unemployment Rate',
    'INDPRO': 'Industrial Production',
    'PAYEMS': 'Nonfarm Payrolls',
    'HOUST': 'Housing Starts',
    'DSPIC96': 'Real Personal Income',
    'PCEC96': 'Real Personal Consumption',
    'GPDI': 'Gross Private Investment',
    'CPILFESL': 'Core CPI',
    'DFF': 'Federal Funds Rate',
    'OPHNFB': 'Labor Productivity',
    'TOTDTEUSQ163N': 'Total Credit',
    'USREC': 'NBER Recession Dates',
    'SP500': 'S&P 500 Index',
}

# NBER recession peak-trough dates (quarterly approximation)
NBER_RECESSIONS = [
    ('1960-04-01', '1961-02-01'),
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01'),
]
