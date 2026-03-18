# Analytics — Bayesian Marketing Mix Model (MMM)

A premier-grade **Marketing Mix Modeling** and **Bayesian Analytics** framework for retail, luxury, CPG, and DTC brands. Built with the same methodological rigor used by **Ekimetrics**, **Analytic Partners**, **Nielsen**, and **Meta's Robyn** — reimplemented with a Bayesian-first approach using **PyMC**.

---

## What This Does

Marketing Mix Modeling quantifies the impact of each marketing channel (TV, digital, social, search, etc.) on business outcomes (revenue, conversions). This framework:

1. **Decomposes revenue** into base, trend, seasonality, and per-channel media contributions
2. **Learns adstock (carry-over)** — how long advertising effects persist after exposure
3. **Learns saturation (diminishing returns)** — the non-linear response to increasing spend
4. **Computes ROI** per channel with full Bayesian uncertainty quantification
5. **Optimizes budget allocation** across channels to maximize expected revenue
6. **Runs scenario analysis** — "what if we increase/decrease total budget by 20%?"

---

## Architecture

```
Analytics/
├── mmm_bayesian/
│   ├── core/               # Config, utilities, scaling, Fourier features
│   │   ├── config.py       # YAML-based model configuration
│   │   └── utils.py        # Scaling, date features, metrics
│   ├── transforms/          # Media transformations
│   │   ├── adstock.py      # Geometric, Weibull, delayed adstock (NumPy + PyTensor)
│   │   └── saturation.py   # Hill, logistic, Gompertz saturation curves
│   ├── models/              # Bayesian model
│   │   └── bayesian_mmm.py # Full PyMC model: build, fit, decompose, validate
│   ├── optimization/        # Budget allocation
│   │   └── budget_optimizer.py  # SLSQP optimizer, response curves, scenarios
│   ├── visualization/       # Publication-quality charts
│   │   └── plots.py        # 8 chart types: waterfall, ROI, response curves, etc.
│   ├── data/                # Data generation
│   │   └── synthetic.py    # Retail, luxury, DTC scenario generators
│   ├── pipeline/            # End-to-end orchestration
│   │   └── runner.py       # Full pipeline: data → model → optimize → report
│   └── cli.py              # Command-line interface
├── examples/                # Ready-to-run examples
├── config_retail.yaml       # Retail brand config
├── config_luxury.yaml       # Luxury brand config
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd Analytics/
pip install -r requirements.txt
```

### 2. Run the Retail Demo (Quick Mode)

```bash
python examples/quick_start.py
```

This generates synthetic 3-year weekly data for a retail brand with 6 channels, fits the Bayesian MMM, and produces all outputs.

### 3. Run via CLI

```bash
# Retail scenario (full MCMC)
python -m mmm_bayesian.cli --scenario retail --output outputs/

# Luxury scenario (quick mode)
python -m mmm_bayesian.cli --scenario luxury --quick

# Custom data
python -m mmm_bayesian.cli --data your_data.csv --config config_retail.yaml
```

### 4. Run Programmatically

```python
from mmm_bayesian.pipeline import run_pipeline

results = run_pipeline(scenario="retail", quick_mode=True)

print(results["metrics"])          # MAPE, RMSE, R², NRMSE
print(results["roi"])              # Per-channel ROI
print(results["optimal_allocation"])  # Budget optimization
```

---

## Industry Scenarios

| Scenario | Channels | Weekly Base Revenue | Key Characteristics |
|----------|----------|--------------------|--------------------|
| **Retail** | TV, Digital Video, Paid Search, Paid Social, Display, Print | $800K | Balanced omnichannel, holiday spikes |
| **Luxury** | Prestige TV, Digital Video, Influencer, Social, Premium OOH, Print Magazine | $2M | Long carry-over, influencer-heavy, summer dip |
| **DTC** | Paid Search, Paid Social, Email, Affiliate, Display Retargeting | $300K | Performance-driven, short carry-over, high growth |

---

## Model Details

### Bayesian Specification

```
y_t = α + trend_t + seasonality_t + Σ_i β_i · Hill(Adstock(x_it)) + Σ_j γ_j · z_jt + ε_t

where:
    α         ~ Normal(0, 2)              # Intercept
    β_i       ~ HalfNormal(0.5)           # Channel coefficients (positive)
    adstock_α ~ Beta(2, 2)                # Carry-over rate
    sat_λ     ~ HalfNormal(1)             # Half-saturation point
    sat_k     ~ Gamma(1, 1)               # Hill steepness
    γ_j       ~ Normal(0, 0.5)            # Control coefficients
    σ         ~ HalfNormal(1)             # Observation noise
```

### Adstock Types

- **Geometric**: `x'_t = x_t + α · x'_{t-1}` — exponential decay, most common
- **Weibull CDF**: Flexible shape from rapid to delayed decay
- **Weibull PDF**: Peak-then-decay pattern for brand campaigns
- **Delayed**: `α^{(lag-θ)²}` — peak effect after θ periods

### Saturation Types

- **Hill**: `x^k / (λ^k + x^k)` — industry standard, interpretable EC50
- **Logistic**: Sigmoid centered at λ
- **Gompertz**: Asymmetric S-curve with minimum effective dose

---

## Outputs

Each pipeline run creates a timestamped output directory with:

```
outputs/run_YYYYMMDD_HHMMSS/
├── plots/
│   ├── decomposition_waterfall.png      # Revenue component waterfall
│   ├── timeseries_decomposition.png     # Stacked area over time
│   ├── roi_comparison.png               # Channel ROI with credible intervals
│   ├── response_curves.png              # Per-channel diminishing returns
│   ├── budget_allocation.png            # Current vs optimal donut charts
│   ├── scenario_analysis.png            # Budget level impact analysis
│   ├── spend_vs_contribution.png        # Efficiency: spend share vs contribution share
│   └── posterior_diagnostics.png        # MCMC trace plots
├── tables/
│   ├── synthetic_data.csv               # Input data
│   ├── parameter_summary.csv            # Posterior parameter estimates
│   ├── decomposition.csv                # Time-series decomposition
│   ├── channel_roi.csv                  # ROI per channel
│   ├── optimal_allocation.csv           # Optimized budget split
│   ├── response_curves.csv              # Response curve data
│   ├── scenario_analysis.csv            # Budget scenario results
│   ├── ground_truth.json                # True parameters (synthetic data)
│   └── validation_metrics.json          # Out-of-sample metrics
├── config.yaml                          # Config used for this run
├── pipeline.log                         # Full execution log
└── results_summary.json                 # Complete results JSON
```

---

## Customization

### Bring Your Own Data

Prepare a CSV with:
- `date` column (weekly granularity recommended)
- `spend_<channel_name>` columns for each media channel
- `revenue` (or your KPI) column
- Optional control variables (e.g., `temperature`, `competitor_price_index`)

Then create a YAML config (see `config_retail.yaml` as template) and run:

```bash
python -m mmm_bayesian.cli --data your_data.csv --config your_config.yaml
```

### Adding New Channels

Add a new entry in the YAML config:

```yaml
channels:
  - name: tiktok
    adstock_type: geometric
    adstock_max_lag: 4
    saturation_type: hill
    min_budget_pct: 0.02
    max_budget_pct: 0.20
```

Ensure your data has a matching `spend_tiktok` column.

---

## References

- **Ekimetrics** — Marketing Mix Modeling methodology
- **Meta Robyn** — Open-source MMM (R/Python)
- **PyMC-Marketing** — Bayesian MMM library
- **Google Meridian** — MMM framework
- Jin, Y., et al. (2017). "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects"
- Chan, D., & Perry, M. (2017). "Challenges and Opportunities in Media Mix Modeling" (Google)
