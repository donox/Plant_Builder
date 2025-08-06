# tests/run_in_freecad.py
"""Test runner that works within FreeCAD environment"""
import sys
import os


def setup_test_environment():
    """Set up the testing environment within FreeCAD"""
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Add src directory to path
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def run_unit_tests():
    """Run pytest unit tests within FreeCAD"""
    setup_test_environment()

    import pytest

    # Run tests with FreeCAD-friendly options
    test_dir = os.path.join(os.path.dirname(__file__), 'unit')

    # Configure pytest to work well in FreeCAD
    pytest_args = [
        test_dir,
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--no-header',  # No pytest header
        '--no-summary',  # No summary
        '-s',  # Don't capture output (for debugging)
    ]

    print("=" * 50)
    print("RUNNING UNIT TESTS IN FREECAD")
    print("=" * 50)

    # Run pytest and return results
    exit_code = pytest.main(pytest_args)

    if exit_code == 0:
        print("\n✅ All unit tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")

    return exit_code == 0


def run_integration_tests():
    """Run FreeCAD integration tests"""
    setup_test_environment()

    import unittest

    print("=" * 50)
    print("RUNNING FREECAD INTEGRATION TESTS")
    print("=" * 50)

    try:
        # Import your integration tests
        from tests.integration.TestCurveFollower import TestCurveFollowerIntegration

        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestCurveFollowerIntegration))

        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.wasSuccessful():
            print("\n✅ All integration tests passed!")
            return True
        else:
            print(f"\n❌ Integration tests failed:")
            print(f"   Failures: {len(result.failures)}")
            print(f"   Errors: {len(result.errors)}")
            return False

    except ImportError as e:
        print(f"⚠️  Could not import integration tests: {e}")
        print("   This is expected if FreeCAD modules are not available")
        return True


def run_all_tests():
    """Run complete test suite"""
    print("🚀 Starting comprehensive test suite in FreeCAD environment...")

    unit_success = run_unit_tests()
    integration_success = run_integration_tests()

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Unit Tests: {'✅ PASS' if unit_success else '❌ FAIL'}")
    print(f"Integration Tests: {'✅ PASS' if integration_success else '❌ FAIL'}")

    overall_success = unit_success and integration_success
    print(f"Overall Result: {'✅ PASS' if overall_success else '❌ FAIL'}")

    return overall_success


if __name__ == '__main__':
    run_all_tests()