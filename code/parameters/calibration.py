"""
Calibration Tables for Structural Models

Maps each structural model to its source paper, parameter values,
literature source values, and notes.
"""

import os
import numpy as np

# Calibration data: model_name -> {source, params: {name: {value, source_value, source_paper, notes}}}
CALIBRATIONS = {
    'PluckingModel': {
        'source': 'Dupraz, Nakamura, Steinsson (2020)',
        'params': {
            'beta': {'value': 0.99, 'source_value': 0.99, 'source_paper': 'DNS (2020)', 'notes': 'Standard quarterly discount factor'},
            'sigma': {'value': 2.0, 'source_value': 2.0, 'source_paper': 'DNS (2020)', 'notes': 'CRRA coefficient'},
            'phi': {'value': 2.0, 'source_value': 2.0, 'source_paper': 'DNS (2020)', 'notes': 'Frisch elasticity inverse'},
            'theta': {'value': 6.0, 'source_value': 6.0, 'source_paper': 'DNS (2020)', 'notes': 'Elasticity of substitution'},
            'rho_a': {'value': 0.9, 'source_value': 0.95, 'source_paper': 'DNS (2020)', 'notes': 'Productivity persistence'},
            'sigma_a': {'value': 0.01, 'source_value': 0.007, 'source_paper': 'DNS (2020)', 'notes': 'Productivity shock std'},
            'rho_d': {'value': 0.8, 'source_value': 0.8, 'source_paper': 'DNS (2020)', 'notes': 'Demand persistence'},
            'sigma_d': {'value': 0.02, 'source_value': 0.02, 'source_paper': 'DNS (2020)', 'notes': 'Demand shock std'},
            'gamma': {'value': 0.0, 'source_value': 0.0, 'source_paper': 'DNS (2020)', 'notes': 'Wage floor = DNWR'},
            'g': {'value': 0.005, 'source_value': 0.005, 'source_paper': 'Standard', 'notes': '2% annual growth'},
        }
    },
    'NetworkModel': {
        'source': 'Baqaee and Farhi (2020)',
        'params': {
            'sigma': {'value': 0.5, 'source_value': 0.5, 'source_paper': 'BF (2020)', 'notes': 'Elasticity of substitution across sectors'},
            'alpha': {'value': 0.5, 'source_value': 0.5, 'source_paper': 'BF (2020)', 'notes': 'Capital share'},
            'network_density': {'value': 0.2, 'source_value': 0.15, 'source_paper': 'BF (2020)', 'notes': 'Input-output network density'},
            'rho_z': {'value': 0.9, 'source_value': 0.95, 'source_paper': 'BF (2020)', 'notes': 'Sectoral productivity persistence'},
            'sigma_z': {'value': 0.02, 'source_value': 0.02, 'source_paper': 'BF (2020)', 'notes': 'Sectoral shock std'},
            'g': {'value': 0.005, 'source_value': 0.005, 'source_paper': 'Standard', 'notes': '2% annual growth'},
        }
    },
    'HANKModel': {
        'source': 'Kaplan, Moll, Violante (2018)',
        'params': {
            'borrowing_limit': {'value': -1.0, 'source_value': -1.0, 'source_paper': 'KMV (2018)', 'notes': 'Natural borrowing limit'},
            'rho_agg': {'value': 0.8, 'source_value': 0.9, 'source_paper': 'KMV (2018)', 'notes': 'Aggregate shock persistence'},
            'sigma_agg': {'value': 0.02, 'source_value': 0.02, 'source_paper': 'KMV (2018)', 'notes': 'Aggregate shock std'},
            'mpc_unconstrained': {'value': 0.1, 'source_value': 0.05, 'source_paper': 'KMV (2018)', 'notes': 'Permanent-income MPC'},
            'mpc_constrained': {'value': 0.8, 'source_value': 0.7, 'source_paper': 'KMV (2018)', 'notes': 'Hand-to-mouth MPC'},
            'g': {'value': 0.005, 'source_value': 0.005, 'source_paper': 'Standard', 'notes': '2% annual growth'},
        }
    },
    'GranularStructuralModel': {
        'source': 'Gabaix (2011)',
        'params': {
            'alpha': {'value': 1.06, 'source_value': 1.06, 'source_paper': 'Gabaix (2011)', 'notes': 'Power law exponent for firm sizes'},
            'sigma_firm': {'value': 0.12, 'source_value': 0.12, 'source_paper': 'Gabaix (2011)', 'notes': 'Firm-level shock std'},
            'rho_firm': {'value': 0.8, 'source_value': 0.85, 'source_paper': 'Gabaix (2011)', 'notes': 'Firm productivity persistence'},
            'entry_rate': {'value': 0.1, 'source_value': 0.1, 'source_paper': 'Gabaix (2011)', 'notes': 'Quarterly entry rate'},
            'exit_threshold': {'value': -2.0, 'source_value': -2.0, 'source_paper': 'Gabaix (2011)', 'notes': 'Productivity exit threshold (std devs)'},
            'g': {'value': 0.005, 'source_value': 0.005, 'source_paper': 'Standard', 'notes': '2% annual growth'},
        }
    },
    'FinancialAcceleratorModel': {
        'source': 'Bernanke, Gertler, Gilchrist (1999)',
        'params': {
            'rho': {'value': 0.8, 'source_value': 0.8, 'source_paper': 'BGG (1999)', 'notes': 'Output gap persistence'},
            'sigma': {'value': 1.0, 'source_value': 1.0, 'source_paper': 'BGG (1999)', 'notes': 'Shock std (reduced form)'},
            'phi': {'value': 0.1, 'source_value': 0.05, 'source_paper': 'BGG (1999)', 'notes': 'Financial wedge sensitivity'},
            'g': {'value': 0.005, 'source_value': 0.005, 'source_paper': 'Standard', 'notes': '2% annual growth'},
        }
    },
    'ZLBModel': {
        'source': 'Eggertsson and Woodford (2003)',
        'params': {
            'sigma': {'value': 1.0, 'source_value': 1.0, 'source_paper': 'EW (2003)', 'notes': 'Intertemporal elasticity'},
            'kappa': {'value': 0.1, 'source_value': 0.024, 'source_paper': 'EW (2003)', 'notes': 'Phillips curve slope'},
            'beta': {'value': 0.99, 'source_value': 0.99, 'source_paper': 'EW (2003)', 'notes': 'Discount factor'},
            'phi_pi': {'value': 1.5, 'source_value': 1.5, 'source_paper': 'Taylor (1993)', 'notes': 'Taylor rule inflation coefficient'},
            'phi_y': {'value': 0.125, 'source_value': 0.125, 'source_paper': 'Taylor (1993)', 'notes': 'Taylor rule output coefficient'},
            'r_natural': {'value': 0.01, 'source_value': 0.01, 'source_paper': 'EW (2003)', 'notes': 'Quarterly natural rate'},
            'sigma_d': {'value': 0.02, 'source_value': 0.02, 'source_paper': 'EW (2003)', 'notes': 'Demand shock std'},
            'rho_d': {'value': 0.8, 'source_value': 0.8, 'source_paper': 'EW (2003)', 'notes': 'Demand shock persistence'},
            'g': {'value': 0.005, 'source_value': 0.005, 'source_paper': 'Standard', 'notes': '2% annual growth'},
        }
    },
    'AsymmetricRBCModel': {
        'source': 'Fernandez-Villaverde et al. (2015)',
        'params': {
            'rho': {'value': 0.9, 'source_value': 0.95, 'source_paper': 'FV et al. (2015)', 'notes': 'TFP persistence'},
            'sigma': {'value': 0.02, 'source_value': 0.007, 'source_paper': 'FV et al. (2015)', 'notes': 'TFP shock std'},
            'skew': {'value': -5, 'source_value': -3, 'source_paper': 'FV et al. (2015)', 'notes': 'Skew-normal shape parameter'},
            'g': {'value': 0.005, 'source_value': 0.005, 'source_paper': 'Standard', 'notes': '2% annual growth'},
        }
    },
    'UncertaintyShocksModel': {
        'source': 'Bloom (2009)',
        'params': {
            'rho_output': {'value': 0.9, 'source_value': 0.95, 'source_paper': 'Bloom (2009)', 'notes': 'Output persistence'},
            'sigma_base': {'value': 0.01, 'source_value': 0.01, 'source_paper': 'Bloom (2009)', 'notes': 'Baseline shock std'},
            'rho_sigma': {'value': 0.7, 'source_value': 0.75, 'source_paper': 'Bloom (2009)', 'notes': 'Uncertainty persistence'},
            'sigma_sigma': {'value': 0.1, 'source_value': 0.1, 'source_paper': 'Bloom (2009)', 'notes': 'Volatility of volatility'},
            'phi': {'value': 0.3, 'source_value': 0.2, 'source_paper': 'Bloom (2009)', 'notes': 'Real options drag coefficient'},
            'g': {'value': 0.005, 'source_value': 0.005, 'source_paper': 'Standard', 'notes': '2% annual growth'},
        }
    },
    'HysteresisModel': {
        'source': 'Blanchard and Summers (1986)',
        'params': {
            'rho': {'value': 0.8, 'source_value': 0.8, 'source_paper': 'BS (1986)', 'notes': 'Cycle persistence'},
            'sigma': {'value': 0.02, 'source_value': 0.02, 'source_paper': 'BS (1986)', 'notes': 'Demand shock std'},
            'delta': {'value': 0.2, 'source_value': '0.1-0.3', 'source_paper': 'BS (1986)', 'notes': 'Hysteresis strength'},
            'g': {'value': 0.005, 'source_value': 0.005, 'source_paper': 'Standard', 'notes': '2% annual growth'},
        }
    },
}


def generate_calibration_latex_table(output_path=None):
    """Generate a LaTeX table of all model calibrations."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'results')
    os.makedirs(output_path, exist_ok=True)

    lines = []
    lines.append(r'\begin{longtable}{llcccc}')
    lines.append(r'\caption{Structural Model Calibrations} \\')
    lines.append(r'\toprule')
    lines.append(r'Model & Parameter & Value & Source Value & Source & Notes \\')
    lines.append(r'\midrule')
    lines.append(r'\endfirsthead')
    lines.append(r'\multicolumn{6}{c}{\textit{(continued)}} \\')
    lines.append(r'\toprule')
    lines.append(r'Model & Parameter & Value & Source Value & Source & Notes \\')
    lines.append(r'\midrule')
    lines.append(r'\endhead')

    for model_name, info in CALIBRATIONS.items():
        first = True
        for param_name, pdata in info['params'].items():
            model_col = info['source'] if first else ''
            sv = pdata['source_value']
            sv_str = str(sv) if not isinstance(sv, float) else f'{sv:.4g}'
            lines.append(
                f"  {model_col} & {param_name} & {pdata['value']:.4g} & {sv_str} & {pdata['source_paper']} & {pdata['notes']} \\\\"
            )
            first = False
        lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{longtable}')

    tex_path = os.path.join(output_path, 'calibration_table.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {tex_path}")
    return tex_path


def validate_calibrations(tolerance=0.5):
    """
    Flag parameters that deviate significantly from literature values.

    tolerance: fractional deviation threshold (0.5 = 50% deviation)
    Returns list of (model, param, our_value, source_value, deviation) tuples.
    """
    flags = []
    for model_name, info in CALIBRATIONS.items():
        for param_name, pdata in info['params'].items():
            source_val = pdata['source_value']
            if isinstance(source_val, str):
                continue  # Skip ranges like '0.1-0.3'
            if source_val == 0:
                continue
            deviation = abs(pdata['value'] - source_val) / abs(source_val)
            if deviation > tolerance:
                flags.append({
                    'model': model_name,
                    'parameter': param_name,
                    'our_value': pdata['value'],
                    'source_value': source_val,
                    'deviation_pct': deviation * 100
                })
    if flags:
        print(f"\nCalibration warnings ({len(flags)} parameters deviate >{tolerance*100:.0f}%):")
        for f in flags:
            print(f"  {f['model']}.{f['parameter']}: {f['our_value']} vs {f['source_value']} ({f['deviation_pct']:.1f}%)")
    else:
        print("All calibrations within tolerance of literature values.")
    return flags


def run_calibration_analysis(output_dir=None):
    """Run calibration validation and generate LaTeX table."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'results')

    print("\n" + "="*60)
    print("CALIBRATION ANALYSIS")
    print("="*60)

    flags = validate_calibrations()
    tex_path = generate_calibration_latex_table(output_dir)

    return {'flags': flags, 'tex_path': tex_path}
