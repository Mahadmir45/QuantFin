"""
Command-line interface for the MMM Bayesian analytics pipeline.

Usage:
    python -m mmm_bayesian.cli --scenario retail --output outputs/
    python -m mmm_bayesian.cli --data data.csv --config config.yaml
    python -m mmm_bayesian.cli --scenario luxury --quick
"""

import argparse
import sys

from .pipeline.runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian Marketing Mix Model — Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scenario retail
  %(prog)s --scenario luxury --quick
  %(prog)s --data weekly_data.csv --config model_config.yaml
  %(prog)s --scenario dtc --output results/
        """,
    )

    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to CSV data file. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML model configuration file.",
    )
    parser.add_argument(
        "--scenario", type=str, default="retail",
        choices=["retail", "luxury", "dtc"],
        help="Synthetic data scenario (default: retail).",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Output directory (default: outputs/).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer MCMC draws for fast iteration.",
    )

    args = parser.parse_args()

    results = run_pipeline(
        data_path=args.data,
        config_path=args.config,
        scenario=args.scenario,
        output_dir=args.output,
        quick_mode=args.quick,
    )

    print(f"\nPipeline complete. Results in: {results['output_dir']}")
    print(f"Validation MAPE: {results['metrics']['MAPE']:.2f}%")
    print(f"Validation R²:   {results['metrics']['R2']:.4f}")


if __name__ == "__main__":
    main()
