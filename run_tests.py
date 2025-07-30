#!/usr/bin/env python3
"""
Test runner script for the evaluation framework.

This script runs all tests and provides a comprehensive test report.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """Run all tests and generate reports."""
    
    print("🧪 Running Document Extraction Evaluation Framework Tests")
    print("=" * 60)
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found. Please install test dependencies:")
        print("   pip install -r requirements-dev.txt")
        return False
    
    # Create test directories if they don't exist
    test_dirs = ["tests/unit", "tests/integration"]
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # Run unit tests
    print("\n📋 Running Unit Tests...")
    unit_result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/unit/", 
        "-v", 
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/unit"
    ], capture_output=True, text=True)
    
    if unit_result.returncode == 0:
        print("✅ Unit tests passed")
    else:
        print("❌ Unit tests failed")
        print(unit_result.stdout)
        print(unit_result.stderr)
    
    # Run integration tests
    print("\n🔗 Running Integration Tests...")
    integration_result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/integration/", 
        "-v", 
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/integration"
    ], capture_output=True, text=True)
    
    if integration_result.returncode == 0:
        print("✅ Integration tests passed")
    else:
        print("❌ Integration tests failed")
        print(integration_result.stdout)
        print(integration_result.stderr)
    
    # Run all tests with coverage
    print("\n📊 Running Complete Test Suite with Coverage...")
    coverage_result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v", 
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/complete",
        "--cov-report=xml:coverage.xml"
    ], capture_output=True, text=True)
    
    if coverage_result.returncode == 0:
        print("✅ All tests passed")
    else:
        print("❌ Some tests failed")
        print(coverage_result.stdout)
        print(coverage_result.stderr)
    
    # Generate test summary
    print("\n📈 Test Summary")
    print("-" * 30)
    
    # Parse test results
    test_output = coverage_result.stdout
    
    # Count test results
    passed = test_output.count("PASSED")
    failed = test_output.count("FAILED")
    skipped = test_output.count("SKIPPED")
    total = passed + failed + skipped
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    # Coverage information
    if "TOTAL" in test_output:
        coverage_lines = [line for line in test_output.split('\n') if "TOTAL" in line]
        if coverage_lines:
            print(f"\nCoverage Report:")
            print(coverage_lines[-1])
    
    # Generate coverage badge
    generate_coverage_badge()
    
    return coverage_result.returncode == 0


def generate_coverage_badge():
    """Generate a coverage badge for the project."""
    
    try:
        import coverage
        
        # Load coverage data
        cov = coverage.Coverage()
        cov.load()
        
        # Calculate total coverage
        total_coverage = cov.report()
        
        # Create badge
        badge_color = "brightgreen" if total_coverage >= 90 else "green" if total_coverage >= 80 else "yellow" if total_coverage >= 70 else "orange" if total_coverage >= 60 else "red"
        
        badge_url = f"https://img.shields.io/badge/coverage-{total_coverage:.1f}%25-{badge_color}"
        
        print(f"\n📊 Coverage Badge: {badge_url}")
        
        # Save badge URL to file
        with open("coverage_badge.txt", "w") as f:
            f.write(badge_url)
            
    except ImportError:
        print("📊 Coverage badge generation skipped (coverage package not available)")
    except Exception as e:
        print(f"📊 Coverage badge generation failed: {e}")


def run_specific_tests(test_path):
    """Run specific tests."""
    
    print(f"🧪 Running tests in: {test_path}")
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        test_path, 
        "-v", 
        "--tb=short"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Tests passed")
    else:
        print("❌ Tests failed")
        print(result.stdout)
        print(result.stderr)
    
    return result.returncode == 0


def main():
    """Main function."""
    
    if len(sys.argv) > 1:
        # Run specific tests
        test_path = sys.argv[1]
        success = run_specific_tests(test_path)
    else:
        # Run all tests
        success = run_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 