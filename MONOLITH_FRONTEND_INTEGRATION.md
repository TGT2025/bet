# Monolith Alpha Engine - Frontend Integration Guide

## Overview
This guide explains how to integrate the Monolith Alpha Engine with your frontend to display trial trades, agent thoughts, strategy breakdowns, and comprehensive iteration summaries.

## What You Can Now See

### 1. Trial Trades (BEFORE Champions!)
View all attempted trades that are being tested before becoming champions:
- Signal details (symbol, action, price, strategy)
- Alpha sources used for each trade
- Execution optimizations applied
- Risk parameters
- Performance metrics
- Status (viable, rejected, pending)

**Endpoint:** `GET /api/monolith/trial_trades?iteration=5&status=viable`

```json
{
  "trial_trades": [
    {
      "timestamp": "2025-11-18T10:30:00",
      "iteration": 5,
      "status": "viable",
      "alpha_sources": ["volatility_risk_premium", "momentum_crash_protection"],
      "signals": [
        {
          "symbol": "btcusdt",
          "action": "BUY",
          "price": 43250.0,
          "strategy": "VolatilityArbitrage",
          "confidence": 0.85
        }
      ],
      "execution_optimizations": ["limit_orders", "atr_stops"],
      "risk_parameters": {"kelly_size": 0.15, "max_dd": 0.12},
      "performance": {"sharpe": 1.8, "win_rate": 0.65}
    }
  ]
}
```

### 2. Detailed Agent Thoughts with Context
See what each agent is thinking and analyzing:
- ResearchQuant analyzing market regimes
- ExecutionQuant optimizing entry/exit
- RiskQuant validating risk limits
- Reasoner creating plans
- Coder implementing strategies

**Endpoint:** `GET /api/monolith/agent_thoughts?agent=ResearchQuant&limit=50`

```json
{
  "thoughts": [
    {
      "timestamp": "2025-11-18T10:30:15",
      "agent": "ResearchQuant",
      "thought": "Analyzing volatility risk premium for current regime",
      "context": {
        "market_regime": "high_volatility",
        "vix": 28.5,
        "recommended_alphas": ["volatility_risk_premium", "gamma_scalping"]
      }
    }
  ]
}
```

### 3. Agent Suggestions for Inter-Agent Communication
View how agents communicate and suggest optimizations:

**Endpoint:** `GET /api/monolith/agent_suggestions?type=entry_optimization&priority=high`

```json
{
  "suggestions": [
    {
      "timestamp": "2025-11-18T10:30:20",
      "agent": "ExecutionQuant",
      "type": "entry_optimization",
      "suggestion": {
        "method": "limit_orders",
        "expected_slippage_reduction": 0.0025
      },
      "priority": "high"
    }
  ]
}
```

### 4. Individual Agent Status Cards
Get detailed metrics for each agent:
- Total actions
- Success/failure counts
- Action breakdown
- Recent activities
- Agent memory state
- Activity patterns

**Endpoint:** `GET /api/monolith/agent_status?agent=ResearchQuant`

```json
{
  "agent": "ResearchQuant",
  "summary": {
    "total_actions": 45,
    "success_rate": 0.82,
    "success_count": 37,
    "failure_count": 8,
    "action_breakdown": {
      "thinking": 20,
      "suggestion": 15,
      "analysis": 10
    }
  },
  "patterns": {
    "time_window_hours": 24,
    "total_activities": 15,
    "most_common_action": "thinking"
  },
  "memory": {
    "tested_alpha_sources": {...},
    "market_regime_history": [...]
  },
  "recent_activities": [...]
}
```

### 5. Strategy Breakdowns with Alpha Sources
Detailed view of each strategy with alpha sources used:

**Endpoint:** `GET /api/monolith/strategy_breakdown?iteration=5`

```json
{
  "strategies": [
    {
      "timestamp": "2025-11-18T10:30:00",
      "iteration": 5,
      "alpha_sources_used": ["volatility_risk_premium", "momentum_crash_protection"],
      "market_regime": "high_volatility",
      "execution_optimizations": ["limit_orders", "atr_stops"],
      "risk_parameters": {"kelly_size": 0.15},
      "performance_metrics": {"sharpe": 1.8, "win_rate": 0.65},
      "trade_count": 28,
      "status": "viable",
      "signals_generated": 42
    }
  ]
}
```

### 6. Comprehensive Iteration Summaries
Full details of each iteration including all phases, agents, and results:

**Endpoint:** `GET /api/monolith/iteration_summary?iteration=5`

```json
{
  "iteration": 5,
  "start_time": "2025-11-18T10:00:00",
  "end_time": "2025-11-18T10:35:00",
  "completed": true,
  "results": {"status": "viable", "alpha_score": 0.42},
  "phases": ["research", "reasoning", "coding", "execution", "risk_validation"],
  "agents_created": ["ResearchQuant", "ExecutionQuant", "RiskQuant"],
  "performance_metrics": {"sharpe": 1.8, "max_dd": 0.12},
  "agent_thoughts": [...],
  "strategies_tested": 3,
  "strategy_details": [...],
  "agent_suggestions": [...],
  "trial_trades": [[...], [...]]
}
```

### 7. Complete Dashboard Data (Efficient Single Call)
Get everything in one API call for dashboard efficiency:

**Endpoint:** `GET /api/monolith/dashboard_data`

```json
{
  "current_iteration": 10,
  "system_status": {
    "champions": 2,
    "errors": 3,
    "uptime_seconds": 3600
  },
  "agents": {
    "ResearchQuant": {...},
    "ExecutionQuant": {...},
    "RiskQuant": {...}
  },
  "recent_thoughts": [...],
  "recent_suggestions": [...],
  "recent_strategies": [...],
  "trial_trades": [...],
  "recommendations": [...]
}
```

## Integration Steps

### Step 1: Import and Initialize in e17final

Add after the existing Flask app setup:

```python
# Import Monolith components
from monolith import get_tracker_instance
from monolith_api_endpoints import register_monolith_endpoints

# Initialize tracker
activity_tracker = get_tracker_instance()

# Register enhanced endpoints
register_monolith_endpoints(app, system, activity_tracker)
```

### Step 2: Use Tracker in Agent Code

When agents perform actions:

```python
# Log agent thinking
activity_tracker.log_agent_thinking(
    agent_name="ResearchQuant",
    thought="Analyzing market regime for alpha selection",
    context={"current_vix": 28.5, "market_state": "high_vol"}
)

# Log agent suggestion
activity_tracker.log_agent_suggestion(
    agent_name="ExecutionQuant",
    suggestion_type="entry_optimization",
    suggestion={"method": "limit_orders", "expected_slippage_reduction": 0.0025},
    priority="high"
)

# Log strategy info
activity_tracker.log_strategy_info(
    iteration=5,
    strategy_data={
        "alpha_sources_used": ["volatility_risk_premium"],
        "signals": [...],
        "execution_optimizations": ["limit_orders"],
        "risk_parameters": {"kelly_size": 0.15},
        "performance_metrics": {"sharpe": 1.8},
        "status": "viable"
    }
)
```

### Step 3: Frontend Implementation

#### React/Vue/Angular Example
```javascript
// Fetch trial trades
async function fetchTrialTrades(iteration) {
  const response = await fetch(`/api/monolith/trial_trades?iteration=${iteration}`);
  const data = await response.json();
  return data.trial_trades;
}

// Fetch complete dashboard data
async function fetchDashboardData() {
  const response = await fetch('/api/monolith/dashboard_data');
  return await response.json();
}

// Fetch agent thoughts
async function fetchAgentThoughts(agentName) {
  const response = await fetch(`/api/monolith/agent_thoughts?agent=${agentName}&limit=100`);
  const data = await response.json();
  return data.thoughts;
}
```

#### HTML/JavaScript Example
```html
<div id="trial-trades"></div>
<script>
  fetch('/api/monolith/trial_trades?limit=50')
    .then(res => res.json())
    .then(data => {
      const container = document.getElementById('trial-trades');
      data.trial_trades.forEach(trade => {
        container.innerHTML += `
          <div class="trade-card">
            <h3>Iteration ${trade.iteration} - ${trade.status}</h3>
            <p>Alpha Sources: ${trade.alpha_sources.join(', ')}</p>
            <p>Signals: ${trade.signals.length}</p>
            <p>Performance: Sharpe ${trade.performance.sharpe}</p>
          </div>
        `;
      });
    });
</script>
```

## Frontend Dashboard Components

### Component 1: Trial Trades Table
Shows all trades being tested before champion selection:
- Iteration number
- Trade signals (symbol, action, price)
- Alpha sources used
- Status (viable/rejected/pending)
- Performance metrics

### Component 2: Agent Activity Timeline
Real-time feed of agent thoughts and actions:
- Agent name with icon
- Thought/action description
- Context details
- Timestamp

### Component 3: Strategy Breakdown Cards
Detailed view of each strategy:
- Alpha sources
- Market regime
- Execution optimizations
- Risk parameters
- Performance metrics
- Trial trade count

### Component 4: Agent Status Dashboard
Grid of agent cards showing:
- Agent name and type
- Success rate
- Total actions
- Last activity
- Current status
- Memory state

### Component 5: Iteration Summary View
Comprehensive iteration details:
- Timeline of phases
- Agents created
- Strategies tested
- Trial trades
- Final results
- Performance metrics

## API Endpoint Reference

| Endpoint | Purpose | Key Data |
|----------|---------|----------|
| `/api/monolith/trial_trades` | View trades before champions | Signals, alpha sources, status |
| `/api/monolith/agent_thoughts` | Agent reasoning process | Thoughts with context |
| `/api/monolith/agent_suggestions` | Inter-agent communication | Suggestions and priorities |
| `/api/monolith/agent_status` | Agent health metrics | Success rates, activities |
| `/api/monolith/strategy_breakdown` | Strategy details | Alpha sources, optimizations |
| `/api/monolith/iteration_summary` | Complete iteration view | All phases and results |
| `/api/monolith/dashboard_data` | Everything in one call | Complete dashboard state |

## Troubleshooting

### "No trial trades showing"
- Check that `activity_tracker.log_strategy_info()` is being called with signal data
- Verify the `signals` field is populated in strategy_data
- Check iteration parameter matches current iteration

### "Agent thoughts not appearing"
- Ensure `activity_tracker.log_agent_thinking()` is called in agent code
- Check agent name matches expected values (ResearchQuant, ExecutionQuant, etc.)
- Verify since_hours parameter if using time filter

### "Endpoints returning 404"
- Confirm `register_monolith_endpoints()` was called in Flask setup
- Check Flask app is running on correct port (default 8000)
- Verify imports are working: `from monolith import get_tracker_instance`

## Performance Considerations

- Use `/api/monolith/dashboard_data` for initial load (one call vs many)
- Implement polling (5-10 second intervals) for real-time updates
- Use query parameters to limit results (default limits in place)
- Cache data client-side and only fetch updates
- Use WebSocket for real-time streaming (future enhancement)

## Next Steps

1. ✅ Import monolith tracker in e17final
2. ✅ Register endpoints with Flask app
3. ✅ Add tracker calls in agent code
4. ✅ Build frontend components
5. ✅ Test with real iterations
6. ✅ Monitor performance and optimize

Your frontend now has complete visibility into the alpha hunting process!
