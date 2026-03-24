# Trend-Cycle Decompositions and Asymmetric Business Cycles

**Bruno Cittolin Smaniotto** | UC Berkeley

## Overview

This paper replaces the quadratic penalty in the Hodrick-Prescott filter with an absolute-value penalty, producing an L1-HP filter that targets the conditional median of the cycle distribution. The resulting trend is piecewise-linear and the extracted cycle is deeper in recessions but largely unchanged during expansions. Monte Carlo experiments show the L1 filter reduces cycle RMSE on asymmetric data-generating processes. Applied to U.S. GDP, the L1 cycle aligns more closely with CBO output gap estimates, implies a flatter Phillips Curve, and prescribes more aggressive monetary easing after deep recessions.

## Quick Start

```bash
pip install -r requirements.txt
python run_replication.py
```

For a quick check (~5 minutes):

```bash
python run_replication.py --quick
```

## Setup

Create `config.json` with your FRED API key (free from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)):

```bash
cp config.json.example config.json
# Edit config.json and add your API key
```

## Replication

The full pipeline (~30-60 minutes) downloads FRED data, runs Monte Carlo experiments on 12 statistical DGPs, simulates 9 structural asymmetry mechanisms, performs empirical analysis on U.S. GDP, and generates all figures and tables.

## Repository Structure

```
run_replication.py              Master pipeline script
code/
  config.py                     Shared constants (lambda values, NBER dates)
  filters/                      L1-HP, L2-HP, Hamilton, Baxter-King, mollified
  models/                       12 statistical DGPs, 9 structural models
  simulations/                  Monte Carlo, structural experiments, sensitivity
  empirical_analysis/           GDP, CBO, Phillips Curve, Taylor Rule, international
  utils/                        Bootstrap, asymmetry tests, specification test
  plotting/                     Figure generation
data/
  raw/                          Downloaded FRED data (not tracked)
  processed/                    Preprocessed quarterly data
output/
  results/                      CSV output tables
  figures/                      PNG figures organized by section
manuscript/
  paper.tex                     LaTeX source
  references.bib                Bibliography
```

## Requirements

- Python 3.10+
- FRED API key
- Dependencies listed in `requirements.txt`

## Citation

```bibtex
@article{smaniotto2026trendcycle,
  title={Trend-Cycle Decompositions and Asymmetric Business Cycles},
  author={Smaniotto, Bruno Cittolin},
  year={2026},
  journal={Working Paper}
}
```

## License

MIT License -- see [LICENSE](LICENSE).
