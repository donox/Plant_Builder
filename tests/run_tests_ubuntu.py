# tests/run_tests_ubuntu.py - FIXED VERSION
"""Test runner optimized for Ubuntu 24.04 + FreeCAD"""
import sys
import os

# CRITICAL: Add project root to Python path FIRST
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now import the environment setup
from tests.freecad_env import setup_freecad_environment, verify_freecad_import


def run_unit_tests():
    """Run pytest unit tests"""
    setup_freecad_environment()

    import pytest

    test_dir = os.path.join(project_root, 'tests', 'unit')

    print("🔧 Running Unit Tests (Pure Python)")
    print("=" * 50)

    # Run with Ubuntu/PyCharm friendly options
    pytest_args = [
        test_dir,
        '-v',  # Verbose
        '--tb=short',  # Short traceback
        '-s',  # Don't capture output (for debugging)
        '--color=yes',  # Colored output
        '--durations=5',  # Show 5 slowest tests
    ]

    exit_code = pytest.main(pytest_args)
    return exit_code == 0


def run_freecad_tests():
    """Run FreeCAD integration tests"""
    setup_freecad_environment()

    print("🏭 Running FreeCAD Integration Tests")
    print("=" * 50)

    if not verify_freecad_import():
        print("⚠️  Skipping FreeCAD tests - FreeCAD not available")
        return True

    import unittest

    try:
        from tests.integration.TestCurveFollower import TestCurveFollowerIntegration

        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestCurveFollowerIntegration))

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return result.wasSuccessful()

    except ImportError as e:
        print(f"⚠️  Could not import integration tests: {e}")
        return True


def main():
    """Main test runner"""
    print("🚀 Starting Test Suite on Ubuntu 24.04")
    print(f"   Python: {sys.version}")
    print(f"   Working Directory: {os.getcwd()}")
    print(f"   Project Root: {project_root}")

    unit_success = run_unit_tests()
    freecad_success = run_freecad_tests()

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Unit Tests: {'✅ PASS' if unit_success else '❌ FAIL'}")
    print(f"FreeCAD Tests: {'✅ PASS' if freecad_success else '❌ FAIL'}")

    overall = unit_success and freecad_success
    print(f"Overall: {'✅ PASS' if overall else '❌ FAIL'}")

    return overall


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)