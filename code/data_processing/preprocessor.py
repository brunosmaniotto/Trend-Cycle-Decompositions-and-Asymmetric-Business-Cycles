import pandas as pd
import numpy as np
import os

def preprocess_fred_data(df, save_processed=True, output_dir='../../data/processed'):
    """
    Preprocess raw FRED data: resample to quarterly, log-transform, etc.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from downloader.
    save_processed : bool
        Whether to save processed data to CSV.
    output_dir : str
        Directory to save processed data.
        
    Returns
    -------
    df_quarterly : pd.DataFrame
        Processed quarterly data.
    """
    # Resample to quarterly frequency (mean)
    df_quarterly = df.resample('Q').mean()
    
    # Log transformations for level variables (not rates)
    # UNRATE and DFF are rates, do not log. USREC is binary.
    log_vars = ['GDPC1', 'INDPRO', 'PAYEMS', 'HOUST', 'DSPIC96', 'PCEC96', 'GPDI', 'CPILFESL', 'OPHNFB', 'TOTDTEUSQ163N', 'SP500']
    
    for var in log_vars:
        if var in df_quarterly.columns:
            df_quarterly[f'{var}_log'] = np.log(df_quarterly[var])
            # Annualized growth rate
            df_quarterly[f'{var}_growth'] = df_quarterly[f'{var}_log'].diff() * 400
            
    # Drop rows only if GDPC1 is missing, to preserve full history for main analysis
    # even if auxiliary variables (like SP500) are missing in early years
    if 'GDPC1_log' in df_quarterly.columns:
        df_quarterly = df_quarterly.dropna(subset=['GDPC1_log'])
    else:
        df_quarterly = df_quarterly.dropna()
    
    if save_processed:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'fred_data_processed.csv')
        df_quarterly.to_csv(output_path)
        print(f"Processed data saved to {output_path}")
        
    return df_quarterly
