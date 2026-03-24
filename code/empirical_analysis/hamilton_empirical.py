"""
Compute Hamilton filter versions of all Section 6 empirical exercises.
Outputs results to output/results/hamilton_empirical/.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from numpy.linalg import lstsq

from filters.hamilton import l1_hamilton_filter, l2_hamilton_filter


def run_hamilton_empirical(base_dir):
    """Run Hamilton filter analogues of all empirical exercises."""

    print("\n" + "=" * 60)
    print("HAMILTON EMPIRICAL: All Section 6 exercises with Hamilton filter")
    print("=" * 60)

    out_dir = os.path.join(base_dir, 'output', 'results', 'hamilton_empirical')
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    fred = pd.read_csv(os.path.join(base_dir, 'data', 'processed',
                                    'fred_data_processed.csv'), parse_dates=['DATE'])
    gdp_results = pd.read_csv(os.path.join(base_dir, 'output', 'results',
                                           'gdp_filter_results.csv'), parse_dates=['Date'])
    gdp_results.rename(columns={'Date': 'DATE'}, inplace=True)
    cbo = pd.read_csv(os.path.join(base_dir, 'output', 'results',
                                   'cbo_comparison_gaps.csv'), parse_dates=['DATE'])

    # 1. CBO Comparison
    print("\n[Hamilton] CBO Comparison...")
    df_cbo = cbo[['DATE', 'Gap_CBO', 'Gap_L2', 'Gap_L1', 'Recession']].copy()
    df_cbo = df_cbo.merge(gdp_results[['DATE', 'Cycle_Ham_L2', 'Cycle_Ham_L1']],
                           on='DATE', how='left')
    df_cbo['Gap_Ham_L2'] = df_cbo['Cycle_Ham_L2'] * 100
    df_cbo['Gap_Ham_L1'] = df_cbo['Cycle_Ham_L1'] * 100
    df_cbo_valid = df_cbo.dropna(subset=['Gap_Ham_L2', 'Gap_Ham_L1', 'Gap_CBO'])

    for label, col in [('CBO', 'Gap_CBO'), ('HP-L2', 'Gap_L2'), ('HP-L1', 'Gap_L1'),
                        ('Ham-L2', 'Gap_Ham_L2'), ('Ham-L1', 'Gap_Ham_L1')]:
        vals = df_cbo_valid[col]
        print(f"  {label:8s}: mean={vals.mean():7.3f}  std={vals.std():6.3f}  "
              f"skew={vals.skew():6.3f}  min={vals.min():7.3f}  max={vals.max():6.3f}")

    for label, col in [('HP-L2', 'Gap_L2'), ('HP-L1', 'Gap_L1'),
                        ('Ham-L2', 'Gap_Ham_L2'), ('Ham-L1', 'Gap_Ham_L1')]:
        corr = df_cbo_valid[col].corr(df_cbo_valid['Gap_CBO'])
        rmse = np.sqrt(np.mean((df_cbo_valid[col] - df_cbo_valid['Gap_CBO'])**2))
        print(f"  {label:8s} vs CBO: corr={corr:.4f}  RMSE={rmse:.4f}")

    df_cbo.to_csv(os.path.join(out_dir, 'hamilton_cbo_comparison.csv'), index=False)

    # 2. Taylor Rule
    print("\n[Hamilton] Taylor Rule...")
    r_star, pi_star = 2.0, 2.0
    df_taylor = fred[['DATE', 'DFF', 'CPILFESL_growth']].copy()
    df_taylor.rename(columns={'CPILFESL_growth': 'inflation'}, inplace=True)
    df_taylor = df_taylor.merge(df_cbo[['DATE', 'Gap_Ham_L2', 'Gap_Ham_L1',
                                         'Gap_L2', 'Gap_L1']], on='DATE', how='inner')
    df_taylor = df_taylor.dropna()

    for gap_col, label in [('Gap_L2', 'HP-L2'), ('Gap_L1', 'HP-L1'),
                             ('Gap_Ham_L2', 'Ham-L2'), ('Gap_Ham_L1', 'Ham-L1')]:
        df_taylor[f'Taylor_{label}'] = (r_star + df_taylor['inflation']
                                         + 0.5 * (df_taylor['inflation'] - pi_star)
                                         + 0.5 * df_taylor[gap_col])

    periods = {
        'Full sample': (None, None),
        'Pre-Volcker (1963-1979)': ('1963-01-01', '1979-12-31'),
        'Volcker-Greenspan (1980-2000)': ('1980-01-01', '2000-12-31'),
        'Post-GFC (2010-2024)': ('2010-01-01', '2024-12-31'),
    }
    print(f"  {'Period':<30s} {'HP-L2':>8s} {'HP-L1':>8s} {'Ham-L2':>8s} {'Ham-L1':>8s}")
    print("  " + "-" * 62)
    for period_name, (start, end) in periods.items():
        mask = pd.Series(True, index=df_taylor.index)
        if start:
            mask &= df_taylor['DATE'] >= start
        if end:
            mask &= df_taylor['DATE'] <= end
        sub = df_taylor[mask]
        corrs = []
        for label in ['HP-L2', 'HP-L1', 'Ham-L2', 'Ham-L1']:
            corr = sub['DFF'].corr(sub[f'Taylor_{label}'])
            corrs.append(corr)
        print(f"  {period_name:<30s} {corrs[0]:8.3f} {corrs[1]:8.3f} {corrs[2]:8.3f} {corrs[3]:8.3f}")

    df_taylor.to_csv(os.path.join(out_dir, 'hamilton_taylor_rule.csv'), index=False)

    # 3. Phillips Curve
    print("\n[Hamilton] Phillips Curve...")
    df_pc = fred[['DATE', 'CPILFESL_growth']].copy()
    df_pc.rename(columns={'CPILFESL_growth': 'inflation'}, inplace=True)
    df_pc = df_pc.merge(df_cbo[['DATE', 'Gap_Ham_L2', 'Gap_Ham_L1',
                                  'Gap_L2', 'Gap_L1']], on='DATE', how='inner')
    df_pc = df_pc.dropna()
    df_pc['inflation_lag'] = df_pc['inflation'].shift(1)
    df_pc = df_pc.dropna()

    results_pc = []
    for gap_col, label in [('Gap_L2', 'HP-L2'), ('Gap_L1', 'HP-L1'),
                             ('Gap_Ham_L2', 'Ham-L2'), ('Gap_Ham_L1', 'Ham-L1')]:
        y = df_pc['inflation'].values
        X = np.column_stack([np.ones(len(y)), df_pc[gap_col].values,
                              df_pc['inflation_lag'].values])
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
        beta = model.params
        se = model.bse
        r2 = model.rsquared
        results_pc.append({
            'Method': label, 'alpha': beta[0], 'beta': beta[1], 'gamma': beta[2],
            'se_beta': se[1], 'r2': r2, 'n_obs': len(y)
        })
        print(f"  {label:8s}: beta={beta[1]:7.4f} (se={se[1]:.4f})  "
              f"gamma={beta[2]:.4f}  R2={r2:.4f}  n={len(y)}")

    pd.DataFrame(results_pc).to_csv(
        os.path.join(out_dir, 'hamilton_phillips_curve.csv'), index=False)

    # 4. Inflation Forecasting
    print("\n[Hamilton] Inflation Forecasting...")
    df_fc = fred[['DATE', 'CPILFESL_growth']].copy()
    df_fc.rename(columns={'CPILFESL_growth': 'inflation'}, inplace=True)
    df_fc = df_fc.merge(df_cbo[['DATE', 'Gap_Ham_L2', 'Gap_Ham_L1',
                                  'Gap_L2', 'Gap_L1']], on='DATE', how='inner')
    df_fc = df_fc.dropna()

    window = 60
    horizons = [1, 4, 8]
    forecast_results = []
    for h in horizons:
        for gap_col, label in [('Gap_L2', 'HP-L2'), ('Gap_L1', 'HP-L1'),
                                 ('Gap_Ham_L2', 'Ham-L2'), ('Gap_Ham_L1', 'Ham-L1'),
                                 (None, 'Naive')]:
            errors = []
            for t in range(window, len(df_fc) - h):
                train = df_fc.iloc[t - window:t]
                actual = df_fc['inflation'].iloc[t + h]
                if label == 'Naive':
                    pred = train['inflation'].iloc[-1]
                else:
                    y_train = train['inflation'].iloc[h:].values
                    X_train = np.column_stack([
                        np.ones(len(y_train)),
                        train[gap_col].iloc[:-h].values,
                        train['inflation'].iloc[:-h].values
                    ])
                    try:
                        beta_fc, _, _, _ = lstsq(X_train, y_train, rcond=None)
                        x_test = np.array([1.0, df_fc[gap_col].iloc[t],
                                            df_fc['inflation'].iloc[t]])
                        pred = x_test @ beta_fc
                    except Exception:
                        pred = train['inflation'].iloc[-1]
                errors.append((actual - pred)**2)

            rmsfe = np.sqrt(np.mean(errors))
            forecast_results.append({
                'Method': label, 'Horizon': h,
                'RMSFE': rmsfe, 'N_forecasts': len(errors)
            })
            print(f"  h={h}  {label:8s}: RMSFE={rmsfe:.4f}  (n={len(errors)})")
        print()

    pd.DataFrame(forecast_results).to_csv(
        os.path.join(out_dir, 'hamilton_inflation_forecast.csv'), index=False)

    # 5. International Evidence
    print("[Hamilton] International Evidence...")
    international_series = {
        'United States': 'GDPC1',
        'United Kingdom': 'CLVMNACSCAB1GQUK',
        'Germany': 'CLVMNACSCAB1GQDE',
        'France': 'CLVMNACSCAB1GQFR',
        'Italy': 'CLVMNACSCAB1GQIT',
        'Japan': 'JPNRGDPEXP',
        'Canada': 'NGDPRSAXDCCAQ',
        'Australia': 'AUSGDPDEFQISMEI',
    }

    config_path = os.path.join(base_dir, 'config.json')
    try:
        with open(config_path) as f:
            FRED_API_KEY = json.load(f)['FRED_API_KEY']
        from fredapi import Fred
        fred_api = Fred(api_key=FRED_API_KEY)
    except Exception as e:
        print(f"  Skipping international (FRED API unavailable): {e}")
        return

    intl_results = []
    for country, series_id in international_series.items():
        try:
            if series_id == 'GDPC1':
                y_raw = fred['GDPC1'].dropna().values
                y_log = np.log(y_raw)
            else:
                raw = fred_api.get_series(series_id)
                raw = raw.dropna()
                y_log = np.log(raw.values.astype(float))

            if len(y_log) < 20:
                print(f"  {country}: too few observations ({len(y_log)}), skipping")
                continue

            try:
                _, cycle_ham_l2 = l2_hamilton_filter(y_log, h=8, p=4)
                _, cycle_ham_l1 = l1_hamilton_filter(y_log, h=8, p=4)
            except Exception as e:
                print(f"  {country}: Hamilton filter failed: {e}")
                continue

            valid = ~(np.isnan(cycle_ham_l2) | np.isnan(cycle_ham_l1))
            c_l2 = cycle_ham_l2[valid]
            c_l1 = cycle_ham_l1[valid]

            if len(c_l2) < 20:
                print(f"  {country}: too few valid Hamilton obs ({len(c_l2)}), skipping")
                continue

            skew_l2 = float(stats.skew(c_l2))
            skew_l1 = float(stats.skew(c_l1))
            std_l2 = float(np.std(c_l2))
            std_l1 = float(np.std(c_l1))
            corr = float(np.corrcoef(c_l2, c_l1)[0, 1])
            _, p_l2 = stats.skewtest(c_l2)
            _, p_l1 = stats.skewtest(c_l1)

            intl_results.append({
                'Country': country, 'N_obs': len(c_l2),
                'Skew_Ham_L2': skew_l2, 'Skew_Ham_L1': skew_l1,
                'Std_Ham_L2': std_l2, 'Std_Ham_L1': std_l1,
                'Skew_p_Ham_L2': p_l2, 'Skew_p_Ham_L1': p_l1,
                'Corr_L1_L2': corr,
            })
            print(f"  {country:15s}: n={len(c_l2):3d}  "
                  f"skew_L2={skew_l2:6.3f} (p={p_l2:.3f})  "
                  f"skew_L1={skew_l1:6.3f} (p={p_l1:.3f})  "
                  f"corr={corr:.3f}")

        except Exception as e:
            print(f"  {country}: failed: {e}")

    pd.DataFrame(intl_results).to_csv(
        os.path.join(out_dir, 'hamilton_international.csv'), index=False)

    print(f"\n  All Hamilton empirical results saved to: {out_dir}")
