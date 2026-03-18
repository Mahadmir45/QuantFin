"""
Quick Start Example — Bayesian Marketing Mix Model
===================================================

This script demonstrates the full MMM pipeline:
    1. Generate synthetic retail data with known ground truth
    2. Configure and fit a Bayesian MMM with PyMC
    3. Decompose revenue by channel
    4. Compute ROI per channel
    5. Optimize budget allocation
    6. Generate all visualizations

Run:
    cd Analytics/
    python examples/quick_start.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mmm_bayesian.pipeline.runner import run_pipeline


def main():
    print("=" * 70)
    print("  BAYESIAN MARKETING MIX MODEL — Quick Start Demo")
    print("  Scenario: Retail Omnichannel (3 years weekly data)")
    print("=" * 70)

    results = run_pipeline(
        scenario="retail",
        output_dir="outputs",
        quick_mode=True,  # Fewer MCMC draws for demo speed
    )

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nValidation Metrics:")
    for k, v in results["metrics"].items():
        print(f"  {k:>8s}: {v:.4f}")

    print(f"\nChannel ROI:")
    for ch in results["roi"]:
        print(f"  {ch['channel']:>20s}: ROI = {ch['roi_mean']:.4f}")

    print(f"\nOptimal Budget Allocation:")
    for ch in results["optimal_allocation"]:
        print(f"  {ch['channel']:>20s}: {ch['optimal_pct']:.1f}% → ROI = {ch['roi']:.4f}")

    print(f"\nAll outputs saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()
