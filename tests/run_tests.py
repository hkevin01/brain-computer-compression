#!/usr/bin/env python3
"""
Comprehensive Test Runner

This script runs all validation and testing suites to ensure the toolkit
is working as intended. It provides different levels of testing based on
user needs and time constraints.
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_command(cmd, description="", timeout=300):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        print(f"\nResult: {'✅ SUCCESS' if success else '❌ FAILED'} (exit code: {result.returncode})")
        
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT after {timeout} seconds")
        return False, "", "Timeout expired"
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, "", str(e)


def run_quick_tests():
    """Run quick unit tests (< 2 minutes)."""
    print("\n🚀 Running QUICK TESTS...")
    
    tests = [
        {
            'cmd': [sys.executable, 'test_simple_validation.py'],
            'description': 'Simple Unit Tests',
            'timeout': 120
        }
    ]
    
    results = []
    for test in tests:
        success, stdout, stderr = run_command(
            test['cmd'], 
            test['description'], 
            test['timeout']
        )
        results.append({
            'name': test['description'],
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        })
    
    return results


def run_standard_tests():
    """Run standard validation tests (< 10 minutes)."""
    print("\n🧪 Running STANDARD TESTS...")
    
    tests = [
        {
            'cmd': [sys.executable, 'test_simple_validation.py'],
            'description': 'Simple Unit Tests',
            'timeout': 120
        },
        {
            'cmd': [sys.executable, 'test_performance_benchmark.py'],
            'description': 'Performance Benchmark',
            'timeout': 300
        }
    ]
    
    results = []
    for test in tests:
        success, stdout, stderr = run_command(
            test['cmd'], 
            test['description'], 
            test['timeout']
        )
        results.append({
            'name': test['description'],
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        })
    
    return results


def run_comprehensive_tests():
    """Run comprehensive validation tests (< 30 minutes)."""
    print("\n🔬 Running COMPREHENSIVE TESTS...")
    
    tests = [
        {
            'cmd': [sys.executable, 'test_simple_validation.py'],
            'description': 'Simple Unit Tests',
            'timeout': 120
        },
        {
            'cmd': [sys.executable, 'test_performance_benchmark.py'],
            'description': 'Performance Benchmark',
            'timeout': 600
        },
        {
            'cmd': [sys.executable, 'test_comprehensive_validation_clean.py'],
            'description': 'Comprehensive Validation',
            'timeout': 900
        }
    ]
    
    results = []
    for test in tests:
        success, stdout, stderr = run_command(
            test['cmd'], 
            test['description'], 
            test['timeout']
        )
        results.append({
            'name': test['description'],
            'success': success,
            'stdout': stdout,
            'stderr': stderr
        })
    
    return results


def check_dependencies():
    """Check if required dependencies are available."""
    print("\n🔍 Checking Dependencies...")
    
    required_packages = [
        'numpy',
        'scipy',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    # Check toolkit modules
    toolkit_modules = [
        'bci_compression.algorithms',
        'bci_compression.algorithms.emg_compression',
        'bci_compression.mobile.emg_mobile',
        'bci_compression.metrics.emg_quality',
        'bci_compression.algorithms.emg_plugins'
    ]
    
    missing_modules = []
    
    for module in toolkit_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_packages or missing_modules:
        print(f"\n⚠️  Missing dependencies:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        for mod in missing_modules:
            print(f"   - {mod}")
        return False
    
    print("\n✅ All dependencies available!")
    return True


def generate_summary_report(test_results, test_level):
    """Generate a summary report of test results."""
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result['success'])
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY REPORT - {test_level.upper()} LEVEL")
    print("="*70)
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print()
    
    # Individual test results
    for result in test_results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} - {result['name']}")
    
    print("\n" + "="*70)
    
    if success_rate >= 80:
        print("🎉 OVERALL STATUS: PASSED")
        print("The toolkit is working as intended!")
    else:
        print("⚠️  OVERALL STATUS: FAILED")
        print("Some components need attention. Review failed tests above.")
    
    print("="*70)
    
    # Save detailed results
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_file = results_dir / f"{test_level}_test_report_{timestamp}.json"
    
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_level': test_level,
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'overall_status': 'PASS' if success_rate >= 80 else 'FAIL'
        },
        'test_results': test_results
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return success_rate >= 80


def run_individual_test(test_name):
    """Run an individual test by name."""
    test_map = {
        'simple': {
            'cmd': [sys.executable, 'test_simple_validation.py'],
            'description': 'Simple Unit Tests',
            'timeout': 120
        },
        'performance': {
            'cmd': [sys.executable, 'test_performance_benchmark.py'],
            'description': 'Performance Benchmark',
            'timeout': 600
        },
        'comprehensive': {
            'cmd': [sys.executable, 'test_comprehensive_validation_clean.py'],
            'description': 'Comprehensive Validation',
            'timeout': 900
        }
    }
    
    if test_name not in test_map:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_map.keys())}")
        return False
    
    test = test_map[test_name]
    success, stdout, stderr = run_command(
        test['cmd'],
        test['description'],
        test['timeout']
    )
    
    return success


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Test Runner for BCI Compression Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Levels:
  quick        - Basic unit tests (~2 minutes)
  standard     - Unit tests + performance benchmarks (~10 minutes)
  comprehensive - All tests including stress tests (~30 minutes)

Individual Tests:
  simple       - Run only simple unit tests
  performance  - Run only performance benchmarks  
  comprehensive - Run only comprehensive validation

Examples:
  python run_tests.py quick
  python run_tests.py standard
  python run_tests.py comprehensive
  python run_tests.py --test simple
  python run_tests.py --dependencies-only
        """
    )
    
    parser.add_argument(
        'level',
        nargs='?',
        choices=['quick', 'standard', 'comprehensive'],
        default='standard',
        help='Test level to run (default: standard)'
    )
    
    parser.add_argument(
        '--test',
        choices=['simple', 'performance', 'comprehensive'],
        help='Run individual test suite'
    )
    
    parser.add_argument(
        '--dependencies-only',
        action='store_true',
        help='Only check dependencies and exit'
    )
    
    parser.add_argument(
        '--no-deps-check',
        action='store_true',
        help='Skip dependency check'
    )
    
    args = parser.parse_args()
    
    print("Brain-Computer Compression Toolkit - Test Runner")
    print("=" * 60)
    
    # Check dependencies first (unless skipped)
    if not args.no_deps_check:
        deps_ok = check_dependencies()
        
        if args.dependencies_only:
            sys.exit(0 if deps_ok else 1)
        
        if not deps_ok:
            print("\n❌ Dependency check failed. Please install missing packages.")
            sys.exit(1)
    
    # Run individual test if specified
    if args.test:
        success = run_individual_test(args.test)
        sys.exit(0 if success else 1)
    
    # Run test suite based on level
    start_time = time.time()
    
    if args.level == 'quick':
        test_results = run_quick_tests()
    elif args.level == 'standard':
        test_results = run_standard_tests()
    elif args.level == 'comprehensive':
        test_results = run_comprehensive_tests()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nTotal test duration: {duration:.1f} seconds")
    
    # Generate summary report
    overall_success = generate_summary_report(test_results, args.level)
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
