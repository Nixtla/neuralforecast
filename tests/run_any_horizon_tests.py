#!/usr/bin/env python3
"""
Test runner script for any horizon predictions functionality.

This script provides an easy way to run the comprehensive test suite
for the any horizon predictions feature.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_tests(test_pattern, verbose=False, coverage=False, parallel=False):
    """Run the specified tests using pytest."""
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=neuralforecast", "--cov-report=term-missing"])
    
    # Add parallel execution if requested
    if parallel:
        cmd.extend(["-n", "auto"])
    
    # Add test pattern
    if test_pattern:
        cmd.append(test_pattern)
    else:
        # Default: run all any horizon related tests
        cmd.append("tests/")
        cmd.extend([
            "-k", "any_horizon or dummy or Dummy",
            "--tb=short"
        ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 80)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False


def run_specific_test_suite(suite_name, verbose=False):
    """Run a specific test suite."""
    
    test_files = {
        "dummy": "tests/test_models/test_dummy_models.py",
        "any_horizon": "tests/test_any_horizon_predictions.py",
        "neuralforecast": "tests/test_neuralforecast_any_horizon.py",
        "exogenous": "tests/test_any_horizon_exogenous.py",
        "all": None
    }
    
    if suite_name not in test_files:
        print(f"❌ Unknown test suite: {suite_name}")
        print(f"Available suites: {', '.join(test_files.keys())}")
        return False
    
    test_file = test_files[suite_name]
    if test_file:
        print(f"🧪 Running {suite_name} test suite: {test_file}")
    else:
        print(f"🧪 Running all test suites")
    
    return run_tests(test_file, verbose=verbose)


def main():
    """Main function to parse arguments and run tests."""
    
    parser = argparse.ArgumentParser(
        description="Run any horizon predictions test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all any horizon tests
  python run_any_horizon_tests.py
  
  # Run specific test suite
  python run_any_horizon_tests.py --suite dummy
  python run_any_horizon_tests.py --suite any_horizon
  
  # Run with verbose output
  python run_any_horizon_tests.py --verbose
  
  # Run with coverage
  python run_any_horizon_tests.py --coverage
  
  # Run specific test file
  python run_any_horizon_tests.py --file tests/test_models/test_dummy_models.py
  
  # Run specific test method
  python run_any_horizon_tests.py --method "test_dummy_univariate_basic"
        """
    )
    
    parser.add_argument(
        "--suite",
        choices=["dummy", "any_horizon", "neuralforecast", "exogenous", "all"],
        help="Run specific test suite"
    )
    
    parser.add_argument(
        "--file",
        help="Run tests from specific file"
    )
    
    parser.add_argument(
        "--method",
        help="Run specific test method"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (skip long-running ones)"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   (where the 'tests' folder is located)")
        sys.exit(1)
    
    # Determine what to run
    if args.suite:
        success = run_specific_test_suite(args.suite, verbose=args.verbose)
    elif args.file:
        test_pattern = args.file
        if args.method:
            test_pattern += f"::{args.method}"
        success = run_tests(test_pattern, verbose=args.verbose, coverage=args.coverage, parallel=args.parallel)
    else:
        # Default: run all any horizon tests
        test_pattern = None
        if args.quick:
            test_pattern = "tests/ -k 'not slow'"
        success = run_tests(test_pattern, verbose=args.verbose, coverage=args.coverage, parallel=args.parallel)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
