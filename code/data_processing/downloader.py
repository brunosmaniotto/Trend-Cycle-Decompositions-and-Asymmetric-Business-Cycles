import pandas as pd
import pandas_datareader as pdr
import os

def download_fred_data(start_date='1960-01-01', end_date='2024-12-31', save_raw=True, output_dir='../../data/raw'):
    """
    Download key U.S. macroeconomic series from FRED.
    
    Parameters
    ----------
    start_date : str
        Start date for download.
    end_date : str
        End date for download.
    save_raw : bool
        Whether to save raw data to CSV.
    output_dir : str
        Directory to save raw data.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing all downloaded series (merged).
    """
    series_dict = {
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
        'SP500': 'S&P 500 Index' # New series
    }
    
    data_frames = []
    
    print("Downloading data from FRED...")
    for series_id, name in series_dict.items():
        try:
            # Fetch data
            series = pdr.get_data_fred(series_id, start=start_date, end=end_date)
            data_frames.append(series)
            print(f"  - {name} ({series_id}): {len(series)} obs")
        except Exception as e:
            print(f"  [ERROR] Failed to download {name} ({series_id}): {e}")
            
    if not data_frames:
        raise RuntimeError("No data downloaded!")
        
    # Merge all series
    df = pd.concat(data_frames, axis=1)
    
    if save_raw:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'fred_data_raw.csv')
        df.to_csv(output_path)
        print(f"Raw data saved to {output_path}")
        
    return df
