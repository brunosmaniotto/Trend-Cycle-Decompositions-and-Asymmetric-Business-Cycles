import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter
from config import LAMBDA_L1_EMPIRICAL, LAMBDA_L2_STANDARD


# Sub-period definitions (inclusive start, exclusive end)
SUB_PERIODS = {
    'Pre-1990':         ('1960-01-01', '1990-01-01'),
    'Great Moderation': ('1990-01-01', '2008-01-01'),
    'Post-GFC':         ('2008-01-01', '2020-01-01'),
    'Post-COVID':       ('2020-01-01', '2099-12-31'),
}


def _estimate_phillips_curve_ols(inflation, gap, lagged_inflation):
    """
    Estimate Phillips Curve via OLS on a single window.

    Model: pi_t = alpha + beta * gap_t + gamma * pi_{t-1} + epsilon_t

    Parameters
    ----------
    inflation : np.ndarray
        Current-period inflation (annualised quarterly rate).
    gap : np.ndarray
        Output gap (percent of potential).
    lagged_inflation : np.ndarray
        One-period lag of inflation.

    Returns
    -------
    results : statsmodels RegressionResultsWrapper or None
        Fitted OLS model, or None if estimation fails.
    """
    X = np.column_stack([gap, lagged_inflation])
    y = inflation

    # Remove observations with NaN in any variable
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    if len(y) < 10:
        return None

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const)
    return model.fit()


def _forecast_inflation(ols_result, gap_value, lagged_inflation_value):
    """
    Produce a one-step Phillips Curve forecast.

    Parameters
    ----------
    ols_result : statsmodels RegressionResultsWrapper
        Fitted model from ``_estimate_phillips_curve_ols``.
    gap_value : float
        Output gap for the forecast period.
    lagged_inflation_value : float
        Lagged inflation for the forecast period.

    Returns
    -------
    float
        Forecasted inflation.
    """
    # params order: [const, beta (gap), gamma (lagged_inflation)]
    return (ols_result.params[0]
            + ols_result.params[1] * gap_value
            + ols_result.params[2] * lagged_inflation_value)


def run_inflation_forecasting(data_path='../../data/processed/fred_data_processed.csv',
                               window=60, horizons=[1, 4, 8]):
    """
    Rolling-window out-of-sample inflation forecasting comparison.

    For each rolling window:
    1. Estimate HP-L1 and HP-L2 on GDP to get output gaps
    2. Estimate Phillips Curve: pi_t = alpha + beta * gap_t + gamma * pi_{t-1}
    3. Forecast inflation h quarters ahead
    4. Compare forecast errors across L1, L2, and naive (random walk)

    Parameters
    ----------
    data_path : str
        Relative path (from this file) to the processed FRED data CSV.
    window : int
        Rolling estimation window in quarters (default 60 = 15 years).
    horizons : list of int
        Forecast horizons in quarters (default [1, 4, 8]).

    Returns
    -------
    rmsfe_df : pd.DataFrame
        RMSFE comparison table by method, horizon, and sub-period.
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
    if not os.path.exists(data_full_path):
        print(f"Data file not found for inflation forecasting: {data_full_path}")
        return None

    df = pd.read_csv(data_full_path, index_col=0, parse_dates=True)

    # GDP in logs (for HP filtering)
    if 'GDPC1_log' not in df.columns:
        print("GDPC1_log column missing. Skipping inflation forecasting.")
        return None

    gdp_log = df['GDPC1_log'].values

    # Core CPI inflation (annualised quarterly growth)
    if 'CPILFESL_growth' in df.columns:
        inflation = df['CPILFESL_growth'].values
    elif 'CPILFESL_log' in df.columns:
        inflation = np.diff(df['CPILFESL_log'].values) * 400
        # Prepend NaN so the series aligns with the original index
        inflation = np.concatenate([[np.nan], inflation])
    else:
        print("CPI data missing. Skipping inflation forecasting.")
        return None

    dates = df.index
    T = len(gdp_log)
    max_h = max(horizons)

    # ------------------------------------------------------------------
    # 2. Rolling-window forecasting
    # ------------------------------------------------------------------
    methods = ['L1-HP', 'L2-HP', 'Naive']
    # Storage: dict of {(method, h): list of (date_index, forecast_error)}
    errors = {(m, h): [] for m in methods for h in horizons}

    n_windows = T - window - max_h
    if n_windows <= 0:
        print(f"Insufficient data: T={T}, window={window}, max_h={max_h}. "
              "Need T > window + max_h.")
        return None

    print(f"Running rolling-window inflation forecasting "
          f"(window={window}, horizons={horizons}, {n_windows} evaluations)...")

    for t_start in range(n_windows):
        t_end = t_start + window  # end of estimation window (exclusive)

        # --- Slice window data ---
        gdp_win = gdp_log[t_start:t_end]
        inf_win = inflation[t_start:t_end]

        # Skip windows that contain NaN inflation at the boundary
        if np.any(np.isnan(inf_win[-2:])):
            continue

        # --- Apply HP filters to GDP within the window ---
        try:
            _, cycle_l1 = l1_hp_filter(gdp_win, lamb=LAMBDA_L1_EMPIRICAL)
        except (RuntimeError, Exception):
            continue
        _, cycle_l2 = l2_hp_filter(gdp_win, lamb=LAMBDA_L2_STANDARD)

        gap_l1 = cycle_l1 * 100  # percent
        gap_l2 = cycle_l2 * 100

        # --- Estimate Phillips Curve on the window ---
        lagged_inf_win = np.concatenate([[np.nan], inf_win[:-1]])

        res_l1 = _estimate_phillips_curve_ols(inf_win, gap_l1, lagged_inf_win)
        res_l2 = _estimate_phillips_curve_ols(inf_win, gap_l2, lagged_inf_win)

        # --- Iterate direct forecasts for each horizon ---
        last_inf = inf_win[-1]

        for h in horizons:
            actual_idx = t_end + h - 1
            if actual_idx >= T or np.isnan(inflation[actual_idx]):
                continue

            actual = inflation[actual_idx]
            forecast_date = dates[actual_idx]

            # Naive forecast: random walk on inflation (last observed value)
            naive_forecast = last_inf
            errors[('Naive', h)].append((forecast_date, actual - naive_forecast))

            # L1-HP Phillips Curve forecast
            if res_l1 is not None:
                # Use end-of-window gap and lagged inflation as predictors
                fc_l1 = _forecast_inflation(res_l1, gap_l1[-1], last_inf)
                errors[('L1-HP', h)].append((forecast_date, actual - fc_l1))

            # L2-HP Phillips Curve forecast
            if res_l2 is not None:
                fc_l2 = _forecast_inflation(res_l2, gap_l2[-1], last_inf)
                errors[('L2-HP', h)].append((forecast_date, actual - fc_l2))

    # ------------------------------------------------------------------
    # 3. Compute RMSFE
    # ------------------------------------------------------------------
    rows = []

    for h in horizons:
        for method in methods:
            err_list = errors[(method, h)]
            if len(err_list) == 0:
                continue
            err_dates, err_vals = zip(*err_list)
            err_arr = np.array(err_vals)
            err_dates = pd.DatetimeIndex(err_dates)

            # Full sample
            rmsfe_full = np.sqrt(np.mean(err_arr ** 2))
            rows.append({
                'Method': method, 'Horizon': h,
                'Period': 'Full Sample',
                'RMSFE': rmsfe_full,
                'N_forecasts': len(err_arr),
            })

            # Sub-periods
            for period_name, (p_start, p_end) in SUB_PERIODS.items():
                mask = (err_dates >= p_start) & (err_dates < p_end)
                if mask.sum() < 4:
                    continue
                rmsfe_sub = np.sqrt(np.mean(err_arr[mask] ** 2))
                rows.append({
                    'Method': method, 'Horizon': h,
                    'Period': period_name,
                    'RMSFE': rmsfe_sub,
                    'N_forecasts': int(mask.sum()),
                })

    rmsfe_df = pd.DataFrame(rows)

    if rmsfe_df.empty:
        print("No valid forecasts produced.")
        return None

    # ------------------------------------------------------------------
    # 4. Save CSV
    # ------------------------------------------------------------------
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'output', 'results'))
    figures_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'output', 'figures'))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, 'inflation_forecast_rmsfe.csv')
    rmsfe_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print summary
    print("\nRMSFE Summary (Full Sample):")
    full_mask = rmsfe_df['Period'] == 'Full Sample'
    pivot = rmsfe_df.loc[full_mask].pivot(index='Horizon', columns='Method', values='RMSFE')
    print(pivot.to_string(float_format='{:.3f}'.format))

    # ------------------------------------------------------------------
    # 5. Generate figure
    # ------------------------------------------------------------------
    _plot_results(errors, rmsfe_df, horizons, methods, figures_dir)

    return rmsfe_df


def _plot_results(errors, rmsfe_df, horizons, methods, figures_dir):
    """
    Generate the inflation forecasting comparison figure.

    Layout:
        Top row    -- 3 panels: cumulative squared forecast error over time
                      (one per horizon)
        Bottom row -- 1 wide panel: bar chart of RMSFE by sub-period

    Parameters
    ----------
    errors : dict
        Mapping of (method, h) -> list of (date, forecast_error).
    rmsfe_df : pd.DataFrame
        RMSFE table produced by ``run_inflation_forecasting``.
    horizons : list of int
        Forecast horizons.
    methods : list of str
        Method labels.
    figures_dir : str
        Directory in which to save the figure.
    """
    style_map = {
        'L1-HP': {'color': '#d62728', 'ls': '-',  'lw': 1.8},
        'L2-HP': {'color': '#1f77b4', 'ls': '--', 'lw': 1.8},
        'Naive': {'color': '#7f7f7f', 'ls': ':',  'lw': 1.4},
    }

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

    # --- Top row: cumulative squared errors ---
    for col_idx, h in enumerate(horizons):
        ax = fig.add_subplot(gs[0, col_idx])
        for method in methods:
            err_list = errors[(method, h)]
            if len(err_list) == 0:
                continue
            err_dates, err_vals = zip(*err_list)
            cum_sq = np.cumsum(np.array(err_vals) ** 2)
            ax.plot(pd.DatetimeIndex(err_dates), cum_sq,
                    label=method, **style_map[method])
        ax.set_title(f'h = {h} quarter{"s" if h > 1 else ""}', fontsize=12)
        ax.set_ylabel('Cumulative Squared Error')
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=30)

    # --- Bottom row: bar chart of RMSFE by sub-period ---
    ax_bar = fig.add_subplot(gs[1, :])

    # Filter to sub-periods only (exclude Full Sample for the bar chart)
    sub_df = rmsfe_df[rmsfe_df['Period'] != 'Full Sample'].copy()
    if sub_df.empty:
        ax_bar.text(0.5, 0.5, 'Insufficient sub-period data',
                    ha='center', va='center', transform=ax_bar.transAxes)
    else:
        # Build grouped bar chart: groups = (Period, Horizon), bars = Method
        period_order = [p for p in SUB_PERIODS.keys()
                        if p in sub_df['Period'].values]
        group_labels = []
        group_data = {m: [] for m in methods}

        for period in period_order:
            for h in horizons:
                label = f'{period}\nh={h}'
                group_labels.append(label)
                for method in methods:
                    row = sub_df[(sub_df['Period'] == period)
                                 & (sub_df['Horizon'] == h)
                                 & (sub_df['Method'] == method)]
                    val = row['RMSFE'].values[0] if len(row) > 0 else 0.0
                    group_data[method].append(val)

        x = np.arange(len(group_labels))
        n_methods = len(methods)
        bar_width = 0.8 / n_methods

        for i, method in enumerate(methods):
            offset = (i - (n_methods - 1) / 2) * bar_width
            ax_bar.bar(x + offset, group_data[method], bar_width,
                       label=method, color=style_map[method]['color'],
                       alpha=0.85, edgecolor='white', linewidth=0.5)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(group_labels, fontsize=8)
        ax_bar.set_ylabel('RMSFE')
        ax_bar.set_title('RMSFE by Sub-Period and Forecast Horizon', fontsize=12)
        ax_bar.legend(fontsize=9)

    fig.suptitle('Out-of-Sample Inflation Forecasting: L1-HP vs L2-HP vs Naive',
                 fontsize=14, fontweight='bold', y=0.98)

    fig_path = os.path.join(figures_dir, 'inflation_forecasting.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_path}")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == '__main__':
    rmsfe = run_inflation_forecasting()
    if rmsfe is not None:
        print("\nDone.")
