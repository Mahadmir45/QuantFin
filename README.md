# QuantFin

Production-grade quantitative finance library: options pricing, portfolio optimization, ML trading strategies, and Bayesian Marketing Mix Modeling.

## Highlights

| Module | Description |
|--------|-------------|
| [Analytics/](Analytics/) | **Bayesian MMM** — PyMC marketing mix modeling with adstock, saturation, ROI, budget optimization |
| [Quant/](Quant/) | Options pricing, portfolio optimization, backtesting, ML strategies (XGBoost, LSTM, Topology Alpha) |
| [Quant_Portfolio/](Quant/Quant_Portfolio/) | TDA + Laplacian diffusion alpha strategy |

## Quick start — Bayesian MMM

```bash
cd Analytics
pip install -r requirements.txt
python -m mmm_bayesian.cli run --config config_retail.yaml --quick
```

## Tech stack

Python 3.10+ · PyMC · NumPy · Pandas · scikit-learn · XGBoost

## Related repos

- [QuantFin-Tableau](https://github.com/Mahadmir45/QuantFin-Tableau) — Executive dashboards
- [ai-engineer-portfolio](https://github.com/Mahadmir45/ai-engineer-portfolio) — Portfolio with RAG demo over QuantFin docs

## Topics

`quantitative-finance` · `bayesian` · `pymc` · `machine-learning` · `marketing-mix-modeling`
