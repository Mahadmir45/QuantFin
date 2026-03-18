#!/usr/bin/env python
"""Test runner for QuantFin Pro."""

import subprocess
import sys
import argparse


def run_tests(test_type='all', verbose=True, coverage=False):
    """
    Run tests with specified options.
    
    Parameters:
    -----------
    test_type : str
        'all', 'unit', 'integration', or specific test file
    verbose : bool
        Verbose output
    coverage : bool
        Generate coverage report
    """
    cmd = ['pytest']
    
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=quantfin', '--cov-report=html', '--cov-report=term'])
    
    if test_type == 'unit':
        cmd.extend(['-m', 'not integration and not slow'])
    elif test_type == 'integration':
        cmd.extend(['-m', 'integration'])
    elif test_type == 'fast':
        cmd.extend(['-m', 'not slow'])
    elif test_type == 'all':
        pass  # Run all tests
    else:
        # Specific test file
        cmd.append(f'tests/test_{test_type}.py')
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run QuantFin Pro tests')
    parser.add_argument(
        'type',
        nargs='?',
        default='all',
        choices=['all', 'unit', 'integration', 'fast', 'options', 'portfolio', 'core', 'strategies'],
        help='Type of tests to run'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Verbose output'
    )
    parser.add_argument(
        '-c', '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    args = parser.parse_args()
    
    return_code = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    sys.exit(return_code)


if __name__ == '__main__':
    main()