"""
Luxury Brand MMM Analysis
=========================

Full pipeline for a luxury/prestige brand scenario with:
    - Prestige TV, influencer, premium OOH, print magazine
    - Longer carry-over effects
    - Higher base revenue and seasonality
    - Influencer channel with delayed adstock

Run:
    cd Analytics/
    python examples/luxury_brand_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmm_bayesian.pipeline.runner import run_pipeline


def main():
    print("=" * 70)
    print("  BAYESIAN MMM — Luxury Brand Analysis")
    print("=" * 70)

    results = run_pipeline(
        scenario="luxury",
        output_dir="outputs_luxury",
        quick_mode=True,
    )

    print(f"\nResults saved to: {results['output_dir']}")
    print(f"Validation R²: {results['metrics']['R2']:.4f}")
    print(f"Validation MAPE: {results['metrics']['MAPE']:.2f}%")


if __name__ == "__main__":
    main()
