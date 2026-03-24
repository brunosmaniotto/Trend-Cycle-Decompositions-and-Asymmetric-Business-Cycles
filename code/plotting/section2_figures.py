"""
Section 2 Figures: The L1-HP Filter

Figure 1: Mean vs Median intuition — asymmetric wave with L2/L1 trend lines
Figure 2: Trend-cycle decomposition — L1 vs L2 on simulated asymmetric data
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from filters.hp import l1_hp_filter, l2_hp_filter

# Base directory: two levels up from code/plotting/ -> Current/
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Output to output/figures/section_2/
_OUTPUT_DIR = os.path.join(_BASE_DIR, 'output', 'figures', 'section_2')


def _make_asymmetric_wave():
    """
    Generate an asymmetric wave: long flat expansions near a ceiling,
    brief sharp plunges.

    Shape per cycle: concave rise (fast recovery, long plateau near ceiling)
    followed by a very short steep drop.
    """
    ceiling = 5.0
    n_points = 600

    # 3 cycles, each with (peak_amp, trough_amp) above/below ceiling.
    # Small peaks above ceiling, varied troughs below => negative skew.
    cycles = [
        (0.6, 2.5),   # moderate recession
        (0.8, 4.0),   # deep recession
        (0.5, 2.8),   # moderate recession
    ]

    x = np.linspace(0, 6 * np.pi, n_points)
    raw_sin = np.sin(x)

    y = np.empty_like(x)
    cycle_len = n_points // 3
    for i, (peak_amp, trough_amp) in enumerate(cycles):
        start = i * cycle_len
        end = (i + 1) * cycle_len if i < 2 else n_points
        seg = raw_sin[start:end]
        # Scale positive and negative parts separately
        y[start:end] = ceiling + np.where(seg >= 0, peak_amp * seg, trough_amp * seg)

    # Add light correlated noise
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 0.08, len(y))
    kernel = np.ones(10) / 10
    noise = np.convolve(noise, kernel, mode='same')
    y = y + noise

    t = np.arange(len(y), dtype=float)
    return t, y


def _draw_bars(ax, t, y, level, color_above, color_below, squared=False):
    """
    Draw residual bars from the trend line to the data.

    If squared=True, bar height = residual^2 (unscaled), showing quadratic penalty.
    If squared=False, bar height = |residual|, showing linear penalty.
    """
    n_bars = 55
    step = max(1, len(t) // n_bars)
    dt = t[1] - t[0]
    bar_width = dt * step * 0.7

    for i in range(0, len(t), step):
        resid = y[i] - level
        if squared:
            h = resid ** 2  # unscaled — shows true quadratic growth
        else:
            h = abs(resid)

        color = color_above if resid >= 0 else color_below
        bottom = level if resid >= 0 else level - h

        rect = plt.Rectangle(
            (t[i] - bar_width / 2, bottom), bar_width, h,
            facecolor=color, alpha=0.3, edgecolor=color, linewidth=0.5
        )
        ax.add_patch(rect)


def figure_01_mean_vs_median(output_dir=None):
    """
    Figure 1: Mean vs Median on asymmetric wave (2x2).

    Top row: clean wave + trend line (what happens).
      Top-left: wave + mean line (L2).
      Top-right: wave + median line (L1).
    Bottom row: residual bars (why it happens).
      Bottom-left: squared residual bars from mean.
      Bottom-right: absolute residual bars from median.
    """
    if output_dir is None:
        output_dir = _OUTPUT_DIR

    t, y = _make_asymmetric_wave()
    y_mean = np.mean(y)
    y_median = np.median(y)

    color_above = '#2166ac'  # blue
    color_below = '#b2182b'  # red
    color_l2 = '#d95f02'     # orange
    color_l1 = '#7570b3'     # purple
    color_wave = '#333333'

    fig, axes = plt.subplots(2, 2, figsize=(13, 8),
                             gridspec_kw={'height_ratios': [1, 1.2]})
    ax_top_l, ax_top_r = axes[0]
    ax_bot_l, ax_bot_r = axes[1]

    # Compute axis limits dynamically
    ylim_top = (-0.8, 6.5)
    # Bottom row: fit the largest squared bar
    max_sq = max((y - y_mean) ** 2)
    ylim_bot = (y_mean - max_sq * 1.05, y_mean + max_sq * 1.05)

    # ===== Top-left: wave + mean =====
    ax_top_l.plot(t, y, color=color_wave, linewidth=1.8, zorder=3)
    ax_top_l.axhline(y_mean, color=color_l2, linewidth=2.5, linestyle='--', zorder=4,
                     label=f'Mean = {y_mean:.1f}')
    ax_top_l.set_title('$p = 2$:  trend targets conditional mean', fontsize=12, fontweight='bold')
    ax_top_l.set_ylabel('$y_t$', fontsize=12)
    ax_top_l.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax_top_l.set_xlim(t[0], t[-1])
    ax_top_l.set_ylim(*ylim_top)
    ax_top_l.tick_params(labelbottom=False)

    # ===== Top-right: wave + median =====
    ax_top_r.plot(t, y, color=color_wave, linewidth=1.8, zorder=3)
    ax_top_r.axhline(y_median, color=color_l1, linewidth=2.5, linestyle='--', zorder=4,
                     label=f'Median = {y_median:.1f}')
    ax_top_r.set_title('$p = 1$:  trend targets conditional median', fontsize=12, fontweight='bold')
    ax_top_r.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax_top_r.set_xlim(t[0], t[-1])
    ax_top_r.set_ylim(*ylim_top)
    ax_top_r.tick_params(labelbottom=False, labelleft=False)

    # ===== Bottom-left: squared residual bars =====
    _draw_bars(ax_bot_l, t, y, y_mean, color_above, color_below, squared=True)
    ax_bot_l.plot(t, y, color=color_wave, linewidth=1.2, alpha=0.5, zorder=3)
    ax_bot_l.axhline(y_mean, color=color_l2, linewidth=2.5, linestyle='--', zorder=4)
    ax_bot_l.set_title('Squared deviations: large residuals dominate', fontsize=11)
    ax_bot_l.set_ylabel('Penalty', fontsize=12)
    ax_bot_l.set_xlabel('Time', fontsize=11)
    ax_bot_l.set_xlim(t[0], t[-1])
    ax_bot_l.set_ylim(*ylim_bot)
    ax_bot_l.tick_params(labelbottom=False)

    # ===== Bottom-right: absolute residual bars =====
    _draw_bars(ax_bot_r, t, y, y_median, color_above, color_below, squared=False)
    ax_bot_r.plot(t, y, color=color_wave, linewidth=1.2, alpha=0.5, zorder=3)
    ax_bot_r.axhline(y_median, color=color_l1, linewidth=2.5, linestyle='--', zorder=4)
    ax_bot_r.set_title('Absolute deviations: each observation votes equally', fontsize=11)
    ax_bot_r.set_xlabel('Time', fontsize=11)
    ax_bot_r.set_xlim(t[0], t[-1])
    ax_bot_r.set_ylim(*ylim_bot)
    ax_bot_r.tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'figure_01_mean_vs_median.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath


def figure_02_trend_cycle_comparison(output_dir=None):
    """
    Figure 2: L1 vs L2 trend-cycle decomposition on simulated asymmetric data.

    Left panel: raw data.
    Middle panel: three trends (true, L2, L1).
    Right panel: three cycles (true, L2, L1).
    """
    if output_dir is None:
        output_dir = _OUTPUT_DIR

    # --- Generate data: linear trend + asymmetric cycle ---
    rng = np.random.RandomState(7)
    T = 200
    t = np.arange(T, dtype=float)

    # Linear trend
    trend_true = 100 + 0.05 * t

    # Asymmetric cycle: same sine construction as figure 1
    # 3 cycles over T observations, asymmetric amplitudes
    cycles_spec = [
        (0.4, 1.8),   # mild recession
        (0.6, 3.5),   # deep recession
        (0.3, 2.0),   # moderate recession
    ]

    x = np.linspace(0, 6 * np.pi, T)
    raw_sin = np.sin(x)
    cycle_true = np.empty(T)
    cycle_len = T // 3
    for i, (peak_amp, trough_amp) in enumerate(cycles_spec):
        start = i * cycle_len
        end = (i + 1) * cycle_len if i < 2 else T
        seg = raw_sin[start:end]
        cycle_true[start:end] = np.where(seg >= 0, peak_amp * seg, trough_amp * seg)

    # Add light noise
    noise = rng.normal(0, 0.15, T)
    kernel = np.ones(5) / 5
    noise = np.convolve(noise, kernel, mode='same')

    y = trend_true + cycle_true + noise

    # --- Apply filters (smoothness-matched lambdas) ---
    trend_l2, cycle_l2 = l2_hp_filter(y, lamb=1600)
    trend_l1, cycle_l1 = l1_hp_filter(y, lamb=130)

    # --- Plot ---
    # Colors + distinct line styles for grayscale readability
    color_l2 = '#d95f02'     # orange
    color_l1 = '#7570b3'     # purple
    color_true = '#1b9e77'   # teal
    color_data = '#666666'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    # Left panel: observed data with true trend
    ax1.plot(t, y, color=color_data, linewidth=1.0, label='Data')
    ax1.plot(t, trend_true, color=color_true, linewidth=2.0, label='True trend')
    ax1.set_title('Simulated data', fontsize=12, fontweight='bold')
    ax1.set_ylabel('$y_t$', fontsize=12)
    ax1.legend(fontsize=9, framealpha=0.9, loc='upper left')
    ax1.tick_params(labelbottom=False)

    # Middle panel: three trends
    # True: solid thick; L2: dashed; L1: dotted thick
    ax2.plot(t, y, color=color_data, linewidth=0.6, alpha=0.25)
    ax2.plot(t, trend_true, color=color_true, linewidth=2.2,
             linestyle='-', label='True trend')
    ax2.plot(t, trend_l2, color=color_l2, linewidth=2.0,
             linestyle='--', label='L2 trend')
    ax2.plot(t, trend_l1, color=color_l1, linewidth=2.5,
             linestyle=':', label='L1 trend')
    ax2.set_title('Extracted trends', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.tick_params(labelbottom=False)

    # Right panel: three cycles
    ax3.plot(t, cycle_true + noise, color=color_true, linewidth=1.5,
             linestyle='-', alpha=0.7, label='True cycle')
    ax3.plot(t, cycle_l2, color=color_l2, linewidth=1.8,
             linestyle='--', label='L2 cycle')
    ax3.plot(t, cycle_l1, color=color_l1, linewidth=2.2,
             linestyle=':', label='L1 cycle')
    ax3.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax3.set_title('Extracted cycles', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cycle', fontsize=12)
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.tick_params(labelbottom=False)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'figure_02_trend_cycle_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath


def figure_03_gdp_investment(output_dir=None):
    """
    Figure 3: L1 vs L2 on US GDP and Investment (2x2).

    Top row: log levels with extracted trends.
    Bottom row: extracted cycles with NBER recession shading.
    Left column: Real GDP. Right column: Gross Private Investment.
    """
    if output_dir is None:
        output_dir = _OUTPUT_DIR

    import pandas as pd
    from config import LAMBDA_L1_EMPIRICAL, LAMBDA_L2_STANDARD, NBER_RECESSIONS

    # Load data
    data_path = os.path.join(_BASE_DIR, 'data', 'processed', 'fred_data_processed.csv')
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    series = {
        'Real GDP': df['GDPC1_log'].dropna(),
        'Gross Private Investment': df['GPDI_log'].dropna(),
    }

    color_l2 = '#d95f02'
    color_l1 = '#7570b3'
    color_data = '#666666'

    # Filter to 2000-2023 window, but run filters on full sample first
    start_date = '2000-01-01'
    end_date = '2023-12-31'

    fig, axes = plt.subplots(2, 2, figsize=(14, 7))

    for row_idx, (label, y_series) in enumerate(series.items()):
        y = y_series.values
        dates = y_series.index

        # Run filters on full sample
        trend_l2, cycle_l2 = l2_hp_filter(y, lamb=LAMBDA_L2_STANDARD)
        trend_l1, cycle_l1 = l1_hp_filter(y, lamb=LAMBDA_L1_EMPIRICAL)

        # Mask to window
        mask = (dates >= start_date) & (dates <= end_date)
        d = dates[mask]
        y_w = y[mask]
        tl2 = trend_l2[mask]
        tl1 = trend_l1[mask]
        cl2 = cycle_l2[mask]
        cl1 = cycle_l1[mask]

        ax_trend = axes[row_idx, 0]
        ax_cycle = axes[row_idx, 1]

        # NBER recession shading
        for peak, trough in NBER_RECESSIONS:
            for ax in [ax_trend, ax_cycle]:
                ax.axvspan(pd.Timestamp(peak), pd.Timestamp(trough),
                           color='grey', alpha=0.2)

        # Left column: levels + trends
        ax_trend.plot(d, y_w, color=color_data, linewidth=0.7, alpha=0.5, label='Data')
        ax_trend.plot(d, tl2, color=color_l2, linewidth=2.2,
                      linestyle='--', label='L2 trend', zorder=3)
        ax_trend.plot(d, tl1, color=color_l1, linewidth=2.5,
                      linestyle='-', label='L1 trend', zorder=4)
        ax_trend.set_ylabel(label, fontsize=12, fontweight='bold')
        if row_idx == 0:
            ax_trend.set_title('Trends', fontsize=13, fontweight='bold')
        ax_trend.legend(fontsize=9, framealpha=0.9, loc='upper left')

        # Right column: cycles
        ax_cycle.plot(d, cl2, color=color_l2, linewidth=1.8,
                      linestyle='--', label='L2 cycle', zorder=3)
        ax_cycle.plot(d, cl1, color=color_l1, linewidth=2.0,
                      linestyle='-', label='L1 cycle', zorder=4)
        ax_cycle.axhline(0, color='black', linewidth=0.8, alpha=0.4)
        if row_idx == 0:
            ax_cycle.set_title('Cycles', fontsize=13, fontweight='bold')
        ax_cycle.legend(fontsize=9, framealpha=0.9, loc='lower left')

        # Enforce x-axis window
        for ax in [ax_trend, ax_cycle]:
            ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'figure_03_gdp_investment.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")
    return filepath


if __name__ == '__main__':
    figure_01_mean_vs_median()
    figure_02_trend_cycle_comparison()
    figure_03_gdp_investment()
