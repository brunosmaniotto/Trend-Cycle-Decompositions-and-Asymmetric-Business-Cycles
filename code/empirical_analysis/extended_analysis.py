import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter
from config import LAMBDA_L1_EMPIRICAL, LAMBDA_L2_STANDARD, HAC_MAXLAGS

def run_extended_analysis(data_path='../../data/processed/fred_data_processed.csv'):
    """
    Run extended empirical analysis:
    3. NAIRU (Unemployment Gap)
    4. Productivity (Labor Productivity Trend)
    5. Business Cycle Dating (vs NBER)
    6. Okun's Law (Gap correlation)
    7. Investment (Asymmetry)
    8. Credit Cycle
    9. Asset Prices (S&P 500)
    """
    # Load data
    data_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
    if not os.path.exists(data_full_path):
        print(f"Data file not found: {data_full_path}")
        return

    df = pd.read_csv(data_full_path, index_col=0, parse_dates=True)
    
    results = {}
    
    # --- 3. NAIRU (Unemployment) ---
    if 'UNRATE' in df.columns:
        u = df['UNRATE'].values
        nairu_l2, u_gap_l2_calc = l2_hp_filter(u, lamb=LAMBDA_L2_STANDARD)
        nairu_l1, u_gap_l1_calc = l1_hp_filter(u, lamb=LAMBDA_L1_EMPIRICAL)
        
        results['NAIRU_L2_Trend'] = nairu_l2
        results['NAIRU_L1_Trend'] = nairu_l1
        results['U_Gap_L2'] = u_gap_l2_calc
        results['U_Gap_L1'] = u_gap_l1_calc
        
        # Calculate Volatility of Trend (Stability)
        vol_nairu_l2 = np.std(np.diff(nairu_l2))
        vol_nairu_l1 = np.std(np.diff(nairu_l1))
        
        results['NAIRU_Volatility'] = {'L2': vol_nairu_l2, 'L1': vol_nairu_l1} # Store here for summary
        print(f"NAIRU Volatility: L2={vol_nairu_l2:.3f}, L1={vol_nairu_l1:.3f}")

    # --- 4. Productivity (OPHNFB) ---
    if 'OPHNFB_log' in df.columns:
        prod = df['OPHNFB_log'].values
        prod_trend_l2, prod_cycle_l2 = l2_hp_filter(prod, lamb=LAMBDA_L2_STANDARD)
        prod_trend_l1, prod_cycle_l1 = l1_hp_filter(prod, lamb=LAMBDA_L1_EMPIRICAL)
        
        results['Prod_Trend_L2'] = prod_trend_l2
        results['Prod_Trend_L1'] = prod_trend_l1
        
        results['Prod_Cycle_L2'] = prod_cycle_l2
        results['Prod_Cycle_L1'] = prod_cycle_l1
        
        print(f"Productivity Trend L2 (Last): {prod_trend_l2[-1]:.3f}, L1 (Last): {prod_trend_l1[-1]:.3f}")

    # --- 5. Business Cycle Dating ---
    if 'USREC' in df.columns and 'GDPC1_log' in df.columns:
        gdp_log = df['GDPC1_log'].values
        _, gdp_cycle_l2 = l2_hp_filter(gdp_log, lamb=LAMBDA_L2_STANDARD)
        _, gdp_cycle_l1 = l1_hp_filter(gdp_log, lamb=LAMBDA_L1_EMPIRICAL)
        
        nber_rec = df['USREC'].values
        
        # Align lengths
        min_len = min(len(nber_rec), len(gdp_cycle_l2), len(gdp_cycle_l1))
        nber_rec_aligned = nber_rec[:min_len]
        gdp_cycle_l2_aligned = gdp_cycle_l2[:min_len]
        gdp_cycle_l1_aligned = gdp_cycle_l1[:min_len]

        # Ensure no NaNs from filter startup
        valid_idx = ~np.isnan(gdp_cycle_l2_aligned) & ~np.isnan(gdp_cycle_l1_aligned)
        
        corr_l2 = np.corrcoef(nber_rec_aligned[valid_idx], gdp_cycle_l2_aligned[valid_idx])[0,1]
        corr_l1 = np.corrcoef(nber_rec_aligned[valid_idx], gdp_cycle_l1_aligned[valid_idx])[0,1]
        
        print(f"NBER Correlation (GDP Cycle): L2={corr_l2:.3f}, L1={corr_l1:.3f}")
        results['NBER_Corr'] = {'L2': corr_l2, 'L1': corr_l1}

    # --- 6. Okun's Law ---
    if 'UNRATE' in df.columns and 'GDPC1_log' in df.columns and 'U_Gap_L2' in results and 'U_Gap_L1' in results:
        # Y Gaps — use date-indexed Series for proper alignment
        gdp_series = df['GDPC1_log'].dropna()
        unrate_series = df['UNRATE'].dropna()
        common_dates = gdp_series.index.intersection(unrate_series.index)

        if len(common_dates) > 20:
            gdp_vals = gdp_series.loc[common_dates].values
            unrate_vals = unrate_series.loc[common_dates].values

            _, y_gap_l2 = l2_hp_filter(gdp_vals, LAMBDA_L2_STANDARD)
            _, y_gap_l1 = l1_hp_filter(gdp_vals, LAMBDA_L1_EMPIRICAL)
            _, u_gap_l2 = l2_hp_filter(unrate_vals, LAMBDA_L2_STANDARD)
            _, u_gap_l1 = l1_hp_filter(unrate_vals, LAMBDA_L1_EMPIRICAL)

            valid = ~(np.isnan(y_gap_l2) | np.isnan(u_gap_l2) | np.isnan(y_gap_l1) | np.isnan(u_gap_l1))

            if valid.sum() > 5:
                # L2 Okun regression with Newey-West SEs
                X_l2 = sm.add_constant(y_gap_l2[valid])
                res_l2 = sm.OLS(u_gap_l2[valid], X_l2).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_MAXLAGS})

                X_l1 = sm.add_constant(y_gap_l1[valid])
                res_l1 = sm.OLS(u_gap_l1[valid], X_l1).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_MAXLAGS})

                print(f"Okun's Law R2: L2={res_l2.rsquared:.3f}, L1={res_l1.rsquared:.3f}")
                results['Okun'] = {
                    'L2_R2': res_l2.rsquared, 'L1_R2': res_l1.rsquared,
                    'L2_Beta': res_l2.params[1], 'L1_Beta': res_l1.params[1],
                    'L2_SE': res_l2.bse[1], 'L1_SE': res_l1.bse[1]
                }
            else:
                print("Not enough valid data points for Okun's Law regression.")
                results['Okun'] = {'L2_R2': np.nan, 'L1_R2': np.nan, 'L2_Beta': np.nan, 'L1_Beta': np.nan}
        else:
            print("Not enough common dates for Okun's Law.")
            results['Okun'] = {'L2_R2': np.nan, 'L1_R2': np.nan, 'L2_Beta': np.nan, 'L1_Beta': np.nan}


    # --- 7. Investment Asymmetry ---
    if 'GPDI_log' in df.columns:
        inv_series = df['GPDI_log'].dropna()
        inv = inv_series.values
        if len(inv) > 20:
            _, inv_cycle_l2 = l2_hp_filter(inv, LAMBDA_L2_STANDARD)
            _, inv_cycle_l1 = l1_hp_filter(inv, LAMBDA_L1_EMPIRICAL)
            
            skew_l2 = stats.skew(inv_cycle_l2)
            skew_l1 = stats.skew(inv_cycle_l1)
            
            print(f"Investment Skewness: L2={skew_l2:.3f}, L1={skew_l1:.3f}")
            results['Inv_Skew'] = {'L2': skew_l2, 'L1': skew_l1}

    # --- 8. Credit Cycle ---
    if 'TOTDTEUSQ163N_log' in df.columns and 'GDPC1_log' in df.columns:
        # Drop NaNs to handle shorter sample
        credit_series = df['TOTDTEUSQ163N_log'].dropna()
        gdp_series = df['GDPC1_log'].dropna()
        
        # Find common index
        common_idx = credit_series.index.intersection(gdp_series.index)
        
        if len(common_idx) > 20:
            credit_clean = credit_series.loc[common_idx].values
            gdp_clean = gdp_series.loc[common_idx].values
            
            _, cred_cycle_l2 = l2_hp_filter(credit_clean, LAMBDA_L2_STANDARD)
            _, cred_cycle_l1 = l1_hp_filter(credit_clean, LAMBDA_L1_EMPIRICAL)

            _, y_cycle_l2 = l2_hp_filter(gdp_clean, LAMBDA_L2_STANDARD)
            _, y_cycle_l1 = l1_hp_filter(gdp_clean, LAMBDA_L1_EMPIRICAL)
            
            # Correlations
            corr_cred_gdp_l2 = np.corrcoef(cred_cycle_l2, y_cycle_l2)[0,1]
            corr_cred_gdp_l1 = np.corrcoef(cred_cycle_l1, y_cycle_l1)[0,1]
            
            print(f"Credit-GDP Correlation: L2={corr_cred_gdp_l2:.3f}, L1={corr_cred_gdp_l1:.3f}")
            results['Credit_Corr'] = {'L2': corr_cred_gdp_l2, 'L1': corr_cred_gdp_l1}

    # --- 9. Asset Prices (S&P 500) ---
    if 'SP500_log' in df.columns:
        sp500_series = df['SP500_log'].dropna()
        sp500 = sp500_series.values
        
        if len(sp500) > 20:
            sp500_trend_l2, sp500_cycle_l2 = l2_hp_filter(sp500, lamb=LAMBDA_L2_STANDARD)
            sp500_trend_l1, sp500_cycle_l1 = l1_hp_filter(sp500, lamb=LAMBDA_L1_EMPIRICAL)

            results['SP500_Cycle_L2'] = sp500_cycle_l2
            results['SP500_Cycle_L1'] = sp500_cycle_l1
            
            sp500_skew_l2 = stats.skew(sp500_cycle_l2)
            sp500_skew_l1 = stats.skew(sp500_cycle_l1)
            print(f"S&P 500 Cycle Skewness: L2={sp500_skew_l2:.3f}, L1={sp500_skew_l1:.3f}")
            results['SP500_Skew'] = {'L2': sp500_skew_l2, 'L1': sp500_skew_l1}
        
    # Save Extended Results

    # Consolidate results into a DataFrame for saving
    summary_data = []

    # NAIRU Volatility
    if 'NAIRU_Volatility' in results:
        summary_data.append({'Metric': 'NAIRU Volatility', 'L2': results['NAIRU_Volatility']['L2'], 'L1': results['NAIRU_Volatility']['L1']})
    # NBER Correlation
    if 'NBER_Corr' in results:
        summary_data.append({'Metric': 'NBER Correlation', 'L2': results['NBER_Corr']['L2'], 'L1': results['NBER_Corr']['L1']})
    # Okun R2
    if 'Okun' in results:
        summary_data.append({'Metric': 'Okun R2', 'L2': results['Okun']['L2_R2'], 'L1': results['Okun']['L1_R2']})
    # Investment Skewness
    if 'Inv_Skew' in results:
        summary_data.append({'Metric': 'Investment Skewness', 'L2': results['Inv_Skew']['L2'], 'L1': results['Inv_Skew']['L1']})
    # Credit-GDP Correlation
    if 'Credit_Corr' in results:
        summary_data.append({'Metric': 'Credit-GDP Correlation', 'L2': results['Credit_Corr']['L2'], 'L1': results['Credit_Corr']['L1']})
    # S&P 500 Skewness
    if 'SP500_Skew' in results:
        summary_data.append({'Metric': 'S&P 500 Skewness', 'L2': results['SP500_Skew']['L2'], 'L1': results['SP500_Skew']['L1']})

    extended_results_df = pd.DataFrame(summary_data)

    script_dir = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'output', 'results'))
    os.makedirs(output_dir, exist_ok=True)

    extended_results_df.to_csv(os.path.join(output_dir, 'extended_analysis_results.csv'), index=False)

    return results
