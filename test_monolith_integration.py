"""
Test script to verify Monolith Alpha Engine integration

This script tests:
1. Import of monolith.agent_activity_tracker
2. Tracker initialization
3. Agent activity logging
4. Memory persistence
5. Data retrieval methods
"""

import sys
import json
from datetime import datetime

print("=" * 80)
print("MONOLITH ALPHA ENGINE - Integration Test")
print("=" * 80)
print()

# Test 1: Import
print("Test 1: Testing import...")
try:
    from monolith.agent_activity_tracker import AgentActivityTracker, get_tracker_instance
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Tracker initialization
print("\nTest 2: Testing tracker initialization...")
try:
    tracker = get_tracker_instance(storage_dir="test_monolith_data")
    print("✅ Tracker initialized")
except Exception as e:
    print(f"❌ Tracker initialization failed: {e}")
    sys.exit(1)

# Test 3: Log agent thinking
print("\nTest 3: Testing agent thinking logging...")
try:
    tracker.log_agent_thinking(
        agent_name="ResearchQuant",
        thought="Analyzing market regime for alpha selection",
        context={"market_regime": "high_volatility", "vix": 28.5}
    )
    print("✅ Agent thinking logged")
except Exception as e:
    print(f"❌ Agent thinking logging failed: {e}")
    sys.exit(1)

# Test 4: Log agent suggestion
print("\nTest 4: Testing agent suggestion logging...")
try:
    tracker.log_agent_suggestion(
        agent_name="ExecutionQuant",
        suggestion_type="entry_optimization",
        suggestion={"method": "limit_orders", "expected_slippage_reduction": 0.0025},
        priority="high"
    )
    print("✅ Agent suggestion logged")
except Exception as e:
    print(f"❌ Agent suggestion logging failed: {e}")
    sys.exit(1)

# Test 5: Log strategy info
print("\nTest 5: Testing strategy info logging...")
try:
    tracker.log_strategy_info(
        iteration=1,
        strategy_data={
            "alpha_sources_used": ["volatility_risk_premium", "momentum_crash_protection"],
            "signals": [
                {"symbol": "btcusdt", "action": "BUY", "price": 43250.0, "confidence": 0.85}
            ],
            "execution_optimizations": ["limit_orders", "atr_stops"],
            "risk_parameters": {"kelly_size": 0.15, "max_dd": 0.12},
            "performance_metrics": {"sharpe": 1.8, "win_rate": 0.65},
            "status": "viable",
            "trade_count": 28
        }
    )
    print("✅ Strategy info logged")
except Exception as e:
    print(f"❌ Strategy info logging failed: {e}")
    sys.exit(1)

# Test 6: Track activity
print("\nTest 6: Testing activity tracking...")
try:
    tracker.track_activity(
        agent_name="RiskQuant",
        action="risk_validation",
        details={"position_size": 0.15, "max_drawdown": 0.12},
        success=True
    )
    print("✅ Activity tracked")
except Exception as e:
    print(f"❌ Activity tracking failed: {e}")
    sys.exit(1)

# Test 7: Get agent summary
print("\nTest 7: Testing agent summary retrieval...")
try:
    summary = tracker.get_agent_summary("ResearchQuant")
    print(f"✅ Agent summary retrieved: {summary['total_actions']} actions")
except Exception as e:
    print(f"❌ Agent summary retrieval failed: {e}")
    sys.exit(1)

# Test 8: Get recent thoughts
print("\nTest 8: Testing recent thoughts retrieval...")
try:
    thoughts = tracker.agent_thinking
    print(f"✅ Retrieved {len(thoughts)} thoughts")
except Exception as e:
    print(f"❌ Thoughts retrieval failed: {e}")
    sys.exit(1)

# Test 9: Get recent suggestions
print("\nTest 9: Testing recent suggestions retrieval...")
try:
    suggestions = tracker.agent_suggestions
    print(f"✅ Retrieved {len(suggestions)} suggestions")
except Exception as e:
    print(f"❌ Suggestions retrieval failed: {e}")
    sys.exit(1)

# Test 10: Get strategy info
print("\nTest 10: Testing strategy info retrieval...")
try:
    strategies = tracker.strategy_info
    print(f"✅ Retrieved {len(strategies)} strategy records")
    if strategies:
        print(f"   First strategy has {len(strategies[0].get('data', {}).get('signals', []))} signals")
except Exception as e:
    print(f"❌ Strategy info retrieval failed: {e}")
    sys.exit(1)

# Test 11: Get adaptive recommendations
print("\nTest 11: Testing adaptive recommendations...")
try:
    recommendations = tracker.get_adaptive_recommendations()
    print(f"✅ Retrieved {len(recommendations)} recommendations")
except Exception as e:
    print(f"❌ Recommendations retrieval failed: {e}")
    sys.exit(1)

# Test 12: Export summary
print("\nTest 12: Testing summary export...")
try:
    summary = tracker.export_summary()
    print(f"✅ Summary exported")
    print(f"   Total activities: {summary['total_activities']}")
    print(f"   Total thoughts: {summary['total_thoughts']}")
    print(f"   Total suggestions: {summary['total_suggestions']}")
    print(f"   Total strategies: {summary['total_strategies']}")
    print(f"   Active agents: {len(summary['agents'])}")
except Exception as e:
    print(f"❌ Summary export failed: {e}")
    sys.exit(1)

# Test 13: Memory persistence
print("\nTest 13: Testing agent memory...")
try:
    tracker.update_agent_memory(
        agent_name="ResearchQuant",
        memory_key="favorite_alpha",
        memory_value="volatility_risk_premium"
    )
    
    memory = tracker.get_agent_memory("ResearchQuant", "favorite_alpha")
    assert memory == "volatility_risk_premium", "Memory value mismatch"
    print("✅ Agent memory working correctly")
except Exception as e:
    print(f"❌ Agent memory test failed: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("ALL TESTS PASSED! ✅")
print("=" * 80)
print()
print("Summary:")
print(f"  • Agents tracked: {len(tracker.agent_stats)}")
print(f"  • Total thoughts: {len(tracker.agent_thinking)}")
print(f"  • Total suggestions: {len(tracker.agent_suggestions)}")
print(f"  • Total strategies: {len(tracker.strategy_info)}")
print(f"  • Total activities: {len(tracker.activities)}")
print()
print("The Monolith Alpha Engine is ready for integration! 🚀")
print()

# Cleanup
import shutil
import os
if os.path.exists("test_monolith_data"):
    shutil.rmtree("test_monolith_data")
    print("Cleaned up test data directory")
