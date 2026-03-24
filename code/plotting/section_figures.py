"""
Publication-quality figures for Sections 5 and 6.
Each function is self-contained and writes directly to output/figures/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from models.structural import (
    PluckingModel, FinancialAcceleratorModel,
    UncertaintyShocksModel, HysteresisModel
)
from filters.hp import l1_hp_filter, l2_hp_filter


def generate_section5_figure(base_dir):
    """
    Section 5 structural model figure (2x2):
    (a) Plucking, (b) Financial Accelerator,
    (c) Uncertainty Shocks, (d) Hysteresis.
    """
    T = 200
    SEEDS = [18, 21, 14, 5]
    PLUCK_WIN = (64, 124)

    C_TRUE = '#333333'
    C_FILL = '#CCCCCC'
    C_L1   = '#2ca02c'
    C_L2   = '#d62728'

    LS_TRUE = '-'
    LS_L1   = '--'
    LS_L2   = ':'

    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes.flatten()

    def cycle_panel(axis, tc, c1, c2, title, window=None):
        sl = slice(window[0], window[1]) if window else slice(None)
        ts = t[sl]
        axis.axhline(0, color='black', lw=1.0, alpha=0.4)
        axis.fill_between(ts, tc[sl], 0, where=(tc[sl] < 0),
                          alpha=0.15, color=C_FILL)
        axis.plot(ts, tc[sl], color=C_TRUE, lw=2.5, ls=LS_TRUE, label='True cycle')
        axis.plot(ts, c2[sl], color=C_L2,   lw=1.5, ls=LS_L2,   label='L2-HP cycle')
        axis.plot(ts, c1[sl], color=C_L1,   lw=1.5, ls=LS_L1,   label='L1-HP cycle')
        axis.set_title(title, fontsize=11, loc='left', pad=5)
        axis.set_ylabel('Cycle', fontsize=9)
        if window:
            axis.set_xlim(window)

    # (a) Plucking
    np.random.seed(SEEDS[0])
    sim_a = PluckingModel().simulate(T=T, burn_in=50)
    _, c1_a = l1_hp_filter(sim_a['output'], lamb=600)
    _, c2_a = l2_hp_filter(sim_a['output'], lamb=1600)
    cycle_panel(ax[0], sim_a['cycle'], c1_a, c2_a,
                '(a) Plucking (capacity ceiling)', window=PLUCK_WIN)

    # (b) Financial Accelerator
    np.random.seed(SEEDS[1])
    sim_b = FinancialAcceleratorModel().simulate(T=T, burn_in=50)
    _, c1_b = l1_hp_filter(sim_b['output'], lamb=600)
    _, c2_b = l2_hp_filter(sim_b['output'], lamb=1600)
    cycle_panel(ax[1], sim_b['cycle'], c1_b, c2_b,
                '(b) Financial Accelerator', window=(50, 150))

    # (c) Uncertainty Shocks
    np.random.seed(SEEDS[2])
    sim_c = UncertaintyShocksModel().simulate(T=T, burn_in=50)
    _, c1_c = l1_hp_filter(sim_c['output'], lamb=600)
    _, c2_c = l2_hp_filter(sim_c['output'], lamb=1600)
    cycle_panel(ax[2], sim_c['cycle'], c1_c, c2_c, '(c) Uncertainty Shocks')

    # (d) Hysteresis
    np.random.seed(SEEDS[3])
    sim_d = HysteresisModel().simulate(T=T, burn_in=50)
    _, c1_d = l1_hp_filter(sim_d['output'], lamb=600)
    _, c2_d = l2_hp_filter(sim_d['output'], lamb=1600)
    cycle_panel(ax[3], sim_d['cycle'], c1_d, c2_d,
                '(d) Hysteresis', window=(50, 150))

    for a in ax:
        a.set_xlabel('Period', fontsize=9)
        a.tick_params(labelsize=8)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    handles = [
        plt.Line2D([0],[0], color=C_TRUE, lw=2.5, ls=LS_TRUE, label='True cycle'),
        plt.Line2D([0],[0], color=C_L1,   lw=1.5, ls=LS_L1,   label='L1-HP cycle ($\\lambda=600$)'),
        plt.Line2D([0],[0], color=C_L2,   lw=1.5, ls=LS_L2,   label='L2-HP cycle ($\\lambda=1600$)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir = os.path.join(base_dir, 'output', 'figures', 'section_5')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'figure_structural_models.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def generate_section6_gdp_figure(base_dir):
    """
    Section 6 figure: US GDP output gap comparison
    (CBO, L1-HP, L2-HP with NBER recession shading).
    """
    df = pd.read_csv(os.path.join(base_dir, 'output', 'results',
                                  'cbo_comparison_gaps.csv'), parse_dates=['DATE'])
    df = df[df['DATE'] >= '1960-01-01'].copy()

    C_CBO = '#333333'
    C_L1  = '#2ca02c'
    C_L2  = '#d62728'
    C_REC = '#DDDDDD'

    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Recession shading
    in_rec = False
    rec_start = None
    for _, row in df.iterrows():
        if row['Recession'] == 1 and not in_rec:
            rec_start = row['DATE']
            in_rec = True
        elif row['Recession'] == 0 and in_rec:
            ax.axvspan(rec_start, row['DATE'], color=C_REC, alpha=0.6, zorder=0)
            in_rec = False
    if in_rec:
        ax.axvspan(rec_start, df['DATE'].iloc[-1], color=C_REC, alpha=0.6, zorder=0)

    ax.axhline(0, color='black', lw=0.8, alpha=0.4)
    ax.plot(df['DATE'], df['Gap_CBO'], color=C_CBO, lw=2.0, ls='-',
            label='CBO output gap')
    ax.plot(df['DATE'], df['Gap_L2'],  color=C_L2,  lw=1.4, ls=':',
            label='L2-HP cycle ($\\lambda=1600$)')
    ax.plot(df['DATE'], df['Gap_L1'],  color=C_L1,  lw=1.4, ls='--',
            label='L1-HP cycle ($\\lambda=600$)')

    ax.set_ylabel('Output gap (\\% of potential GDP)', fontsize=10)
    ax.set_xlabel('')
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(df['DATE'].iloc[0], df['DATE'].iloc[-1])

    ax.legend(loc='lower left', fontsize=9, frameon=False,
              bbox_to_anchor=(0.0, 0.08))
    ax.text(0.01, 0.02, 'Grey shading: NBER recessions', transform=ax.transAxes,
            fontsize=8, color='#666666', va='bottom', ha='left')

    fig.tight_layout()

    out_dir = os.path.join(base_dir, 'output', 'figures', 'section_6')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'figure_gdp_cbo.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def generate_section6_taylor_figure(base_dir):
    """
    Section 6 figure: Taylor Rule counterfactual (2x1).
    Top: output gaps. Bottom: interest rates.
    """
    cbo = pd.read_csv(os.path.join(base_dir, 'output', 'results',
                                   'cbo_comparison_gaps.csv'), parse_dates=['DATE'])
    pol = pd.read_csv(os.path.join(base_dir, 'output', 'results',
                                   'policy_paths.csv'), parse_dates=['DATE'])
    fred = pd.read_csv(os.path.join(base_dir, 'data', 'processed',
                                    'fred_data_processed.csv'), parse_dates=['DATE'])
    fred = fred[['DATE', 'DFF']].dropna()

    df = cbo[['DATE', 'Gap_CBO', 'Gap_L2', 'Gap_L1', 'Recession']].copy()
    df = df.merge(pol[['DATE', 'Taylor_L2', 'Taylor_L1']], on='DATE', how='left')
    df = df.merge(fred, on='DATE', how='left')
    df = df[df['DATE'] >= '1960-01-01'].copy()

    C_CBO = '#333333'
    C_L1  = '#2ca02c'
    C_L2  = '#d62728'
    C_REC = '#DDDDDD'
    C_FFR = '#333333'

    fig, (ax_gap, ax_rate) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    def add_recession_shading(axis, data):
        in_rec = False
        rec_start = None
        for _, row in data.iterrows():
            if row['Recession'] == 1 and not in_rec:
                rec_start = row['DATE']
                in_rec = True
            elif row['Recession'] == 0 and in_rec:
                axis.axvspan(rec_start, row['DATE'], color=C_REC, alpha=0.6, zorder=0)
                in_rec = False
        if in_rec:
            axis.axvspan(rec_start, data['DATE'].iloc[-1], color=C_REC, alpha=0.6, zorder=0)

    add_recession_shading(ax_gap, df)
    add_recession_shading(ax_rate, df)

    # Top panel: Output gaps
    ax_gap.axhline(0, color='black', lw=0.8, alpha=0.4)
    ax_gap.plot(df['DATE'], df['Gap_CBO'], color=C_CBO, lw=2.0, ls='-',
                label='CBO output gap')
    ax_gap.plot(df['DATE'], df['Gap_L2'],  color=C_L2,  lw=1.4, ls=':',
                label='L2-HP cycle ($\\lambda=1600$)')
    ax_gap.plot(df['DATE'], df['Gap_L1'],  color=C_L1,  lw=1.4, ls='--',
                label='L1-HP cycle ($\\lambda=600$)')

    ax_gap.set_ylabel('Output gap (% of potential)', fontsize=10)
    ax_gap.legend(loc='lower left', fontsize=9, frameon=False)
    ax_gap.spines['top'].set_visible(False)
    ax_gap.spines['right'].set_visible(False)
    ax_gap.tick_params(labelsize=9)
    ax_gap.set_title('(a) Output gaps', fontsize=11, loc='left', pad=5)

    # Bottom panel: Interest rates
    ax_rate.axhline(0, color='black', lw=0.8, alpha=0.4)
    ax_rate.plot(df['DATE'], df['DFF'],       color=C_FFR, lw=2.0, ls='-',
                 label='Actual federal funds rate')
    ax_rate.plot(df['DATE'], df['Taylor_L2'], color=C_L2,  lw=1.4, ls=':',
                 label='Taylor Rule (L2 gap)')
    ax_rate.plot(df['DATE'], df['Taylor_L1'], color=C_L1,  lw=1.4, ls='--',
                 label='Taylor Rule (L1 gap)')

    ax_rate.set_ylabel('Interest rate (%)', fontsize=10)
    ax_rate.set_xlabel('')
    ax_rate.legend(loc='upper left', fontsize=9, frameon=False)
    ax_rate.spines['top'].set_visible(False)
    ax_rate.spines['right'].set_visible(False)
    ax_rate.tick_params(labelsize=9)
    ax_rate.set_title('(b) Taylor Rule prescriptions', fontsize=11, loc='left', pad=5)

    ax_rate.xaxis.set_major_locator(mdates.YearLocator(10))
    ax_rate.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_rate.set_xlim(df['DATE'].iloc[0], df['DATE'].iloc[-1])
    ax_rate.text(0.01, 0.02, 'Grey shading: NBER recessions', transform=ax_rate.transAxes,
                 fontsize=8, color='#666666', va='bottom', ha='left')

    fig.tight_layout()

    out_dir = os.path.join(base_dir, 'output', 'figures', 'section_6')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'figure_taylor_rule.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
