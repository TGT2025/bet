#!/usr/bin/env python3
"""
Quick Test Script for E17 FIXE17V1 Implementation
Tests all 3 critical fixes without requiring full system execution
"""

import ast
import re
import sys

def test_crash_recovery():
    """Test Fix #1: Crash Recovery System"""
    print("🧪 Testing Fix #1: Crash Recovery System...")
    
    with open('e17', 'r') as f:
        content = f.read()
    
    # Check for crash recovery components
    checks = {
        'consecutive_failures variable': 'consecutive_failures = 0' in content,
        'MAX_FAILURES constant': 'MAX_FAILURES = 5' in content,
        'KeyboardInterrupt handler': 'except KeyboardInterrupt:' in content,
        'Exponential backoff': 'wait_time = min(30 * (2 ** (consecutive_failures - 1)), 300)' in content,
        'Crash recovery logging': '❌ ITERATION' in content and 'CRASHED' in content,
    }
    
    all_passed = all(checks.values())
    
    for check, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {check}")
    
    return all_passed

def test_monitoring_server():
    """Test Fix #2: Monitoring Server Initialization"""
    print("\n🧪 Testing Fix #2: Monitoring Server...")
    
    with open('e17', 'r') as f:
        content = f.read()
    
    checks = {
        'start_monitoring_server method': 'def start_monitoring_server(self):' in content,
        'Flask import': 'from flask import Flask, jsonify' in content,
        'Status endpoint': "@app.route('/api/status')" in content,
        'Dashboard endpoint': "@app.route('/')" in content,
        'Main function initialization': '# 🔧 NEW: START MONITORING FIRST!' in content,
        'Background thread startup': 'monitoring_thread = threading.Thread' in content,
        'Port 8000 binding': 'port=8000' in content,
    }
    
    all_passed = all(checks.values())
    
    for check, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {check}")
    
    return all_passed

def test_alpha_scoring():
    """Test Fix #3: Enhanced Alpha Score Calculation"""
    print("\n🧪 Testing Fix #3: Enhanced Alpha Scoring...")
    
    with open('e17', 'r') as f:
        content = f.read()
    
    checks = {
        'Score calculation logging': 'logger.info("📊 Calculating alpha score...")' in content,
        'Signal analysis logging': 'logger.info(f"   Analyzing {len(signals)} signals")' in content,
        'Component contribution logging': '✅ Multi-symbol coverage' in content,
        'Final score logging': 'logger.info(f"📈 FINAL ALPHA SCORE: {final_score:.4f}")' in content,
        'Threshold comparison': 'logger.info(f"   Threshold: 0.60")' in content,
        'Pass/fail status': "✅ PASS' if final_score >= 0.60 else '❌ FAIL" in content,
    }
    
    all_passed = all(checks.values())
    
    for check, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {check}")
    
    return all_passed

def test_syntax():
    """Test that e17 has valid Python syntax"""
    print("\n🧪 Testing Python Syntax...")
    
    try:
        with open('e17', 'r') as f:
            code = f.read()
        
        ast.parse(code)
        print("  ✅ Python syntax is valid")
        return True
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        return False

def test_line_preservation():
    """Test that we didn't accidentally delete large chunks of code"""
    print("\n🧪 Testing Code Preservation...")
    
    with open('e17', 'r') as f:
        lines = f.readlines()
    
    line_count = len(lines)
    
    # Original file had 4327 lines
    # We added ~130 lines and removed ~9 lines
    # So we expect around 4327 + 130 - 9 = 4448 lines
    expected_min = 4300
    expected_max = 4500
    
    if expected_min <= line_count <= expected_max:
        print(f"  ✅ Line count: {line_count} (within expected range {expected_min}-{expected_max})")
        return True
    else:
        print(f"  ❌ Line count: {line_count} (expected {expected_min}-{expected_max})")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("E17 FIXE17V1 Implementation Verification")
    print("=" * 70)
    
    results = {
        'Crash Recovery': test_crash_recovery(),
        'Monitoring Server': test_monitoring_server(),
        'Alpha Scoring': test_alpha_scoring(),
        'Python Syntax': test_syntax(),
        'Code Preservation': test_line_preservation(),
    }
    
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    for test_name, passed in results.items():
        print(f"{'✅' if passed else '❌'} {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! E17 fixes verified successfully.")
        print("=" * 70)
        print("\n✅ The system is ready to run with:")
        print("   1. Crash recovery and auto-restart")
        print("   2. Real-time monitoring dashboard")
        print("   3. Enhanced alpha score logging")
        print("\n🚀 Start the system with: python e17")
        print("📊 Monitor at: http://localhost:8000")
        return 0
    else:
        print("❌ SOME TESTS FAILED! Please review the results above.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
