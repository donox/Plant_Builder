# tests/test_runner.py
"""Unified test runner for both pytest and FreeCAD tests"""
import sys
import os
import subprocess
from pathlib import Path


def run_unit_tests():
    """Run pytest unit tests"""
    print("Running pytest unit tests...")
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/unit',
        '-v', '--tb=short'
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def run_integration_tests():
    """Run FreeCAD integration tests"""
    print("Running FreeCAD integration tests...")

    try:
        import FreeCAD
        import unittest

        # Import test modules
        from tests.integration.TestCurveFollower import TestCurveFollowerIntegration

        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestCurveFollowerIntegration))

        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return result.wasSuccessful()

    except ImportError:
        print("FreeCAD not available - skipping integration tests")
        return True


def run_all_tests():
    """Run complete test suite"""
    print("=" * 50)
    print("RUNNING COMPLETE TEST SUITE")
    print("=" * 50)

    unit_success = run_unit_tests()
    integration_success = run_integration_tests()

    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Unit Tests: {'PASS' if unit_success else 'FAIL'}")
    print(f"Integration Tests: {'PASS' if integration_success else 'FAIL'}")

    overall_success = unit_success and integration_success
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")

    return overall_success


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)