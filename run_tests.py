#!/usr/bin/env python3
"""
Test runner for the data_to_concept_unlearning project

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Run with verbose output
    python run_tests.py --pipeline   # Run only pipeline tests
    python run_tests.py --core       # Run only core function tests
"""

import sys
import unittest
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_all_tests(verbosity=1):
    """Run all tests in the tests directory"""
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_pipeline_tests(verbosity=1):
    """Run only pipeline tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_ripple_bench_split')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_core_tests(verbosity=1):
    """Run only core function tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_ripple_bench_core')
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Run tests for data_to_concept_unlearning")
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Run tests with verbose output')
    parser.add_argument('--pipeline', action='store_true',
                       help='Run only pipeline tests')
    parser.add_argument('--core', action='store_true',
                       help='Run only core function tests')
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    
    print(f"Running tests from {project_root}")
    print("=" * 70)
    
    if args.pipeline:
        print("Running pipeline tests...")
        success = run_pipeline_tests(verbosity)
    elif args.core:
        print("Running core function tests...")
        success = run_core_tests(verbosity)
    else:
        print("Running all tests...")
        success = run_all_tests(verbosity)
    
    print("=" * 70)
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()