# Frontend Integration Guide - Agent Activity Dashboard

## Complete Frontend Code Structure

This guide provides pure code (no design) for frontend developers to integrate with the Monolith Alpha Engine backend.

---

## 1. API Endpoint Structure

### Main Dashboard Data Endpoint
```javascript
// GET /api/agent-dashboard
// Returns: Complete dashboard data in single call

const getDashboardData = async () => {
  const response = await fetch('/api/agent-dashboard');
  const data = await response.json();
  
  return data; // Structure defined below
};
```

### Response JSON Structure
```json
{
  "timestamp": "2025-11-17T21:29:22.027Z",
  "agent_status": {
    "ResearchQuant": {
      "last_active": "2025-11-17T21:28:15.123Z",
      "last_thought": "Analyzing volatility risk premium alpha source",
      "activity_count": 45
    },
    "ExecutionQuant": {
      "last_active": "2025-11-17T21:28:18.456Z",
      "last_thought": "Optimizing entry logic for momentum strategy",
      "activity_count": 38
    },
    "RiskQuant": {
      "last_active": "2025-11-17T21:28:20.789Z",
      "last_thought": "Validating position sizes with Kelly criterion",
      "activity_count": 32
    },
    "Reasoner": {
      "last_active": "2025-11-17T21:28:12.012Z",
      "last_thought": "Creating execution plan based on researched alphas",
      "activity_count": 52
    },
    "Coder": {
      "last_active": "2025-11-17T21:28:25.345Z",
      "last_thought": "Implementing strategies.py with optimized execution",
      "activity_count": 41
    }
  },
  "recent_activities": [
    {
      "timestamp": "2025-11-17T21:28:25.345Z",
      "unix_time": 1700255305.345,
      "agent": "Coder",
      "thought": "Implementing strategies.py with optimized execution",
      "context": {
        "file": "strategies.py",
        "lines_written": 250,
        "alpha_sources_implemented": ["volatility_risk_premium", "momentum_crash_protection"]
      },
      "type": "thinking"
    },
    {
      "timestamp": "2025-11-17T21:28:20.789Z",
      "unix_time": 1700255300.789,
      "agent": "RiskQuant",
      "suggestion_type": "position_sizing",
      "suggestion": {
        "kelly_size": 0.15,
        "risk_parity_size": 0.12,
        "vol_target_size": 0.18,
        "recommended": "vol_target_size",
        "reason": "Current volatility regime favors volatility targeting"
      },
      "type": "suggestion"
    }
  ],
  "latest_strategy": {
    "timestamp": "2025-11-17T21:28:00.000Z",
    "unix_time": 1700255280.0,
    "iteration": 5,
    "strategy_data": {
      "alpha_sources_used": [
        {
          "name": "Volatility Risk Premium",
          "expected_sharpe": 1.2,
          "implementation_status": "implemented"
        },
        {
          "name": "Momentum Crash Protection",
          "expected_sharpe": 0.8,
          "implementation_status": "implemented"
        }
      ],
      "execution_optimizations_applied": [
        "limit_order_entry",
        "atr_based_stops",
        "volume_weighted_execution"
      ],
      "risk_parameters": {
        "max_drawdown": 0.15,
        "var_95": 0.02,
        "position_size_method": "volatility_targeting"
      },
      "expected_performance": {
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.12,
        "win_rate": 0.58
      },
      "code_files_generated": [
        "strategies.py",
        "alpha_orchestrator.py",
        "research_quant_agent.py",
        "execution_quant_agent.py",
        "risk_quant_agent.py"
      ]
    },
    "type": "strategy_info"
  },
  "latest_iteration": {
    "timestamp": "2025-11-17T21:28:00.000Z",
    "unix_time": 1700255280.0,
    "iteration": 5,
    "summary": {
      "research_agent_inputs": [
        "Volatility Risk Premium (Sharpe: 1.2)",
        "Momentum Crash Protection (Sharpe: 0.8)"
      ],
      "execution_agent_optimizations": [
        "Limit orders at mid-price",
        "ATR-based trailing stops",
        "Volume-weighted execution"
      ],
      "risk_agent_validations": [
        "Position sizing: Kelly criterion",
        "Risk limits: Max DD 15%, VaR 2%",
        "Stress test: Passed"
      ],
      "reasoner_decisions": [
        "Implement volatility-based strategies",
        "Focus on crash protection during high VIX"
      ],
      "coder_outputs": [
        "strategies.py (250 lines)",
        "alpha_orchestrator.py (180 lines)",
        "3 specialized agents"
      ],
      "final_results": {
        "status": "viable",
        "alpha_score": 0.62,
        "total_trades": 24,
        "diversity": 0.7,
        "avg_confidence": 0.65
      }
    },
    "type": "iteration_summary"
  },
  "agent_thoughts_summary": {
    "ResearchQuant": [
      {
        "timestamp": "2025-11-17T21:28:15.123Z",
        "thought": "Analyzing volatility risk premium alpha source",
        "context": {"market_regime": "high_volatility"}
      }
    ],
    "ExecutionQuant": [
      {
        "timestamp": "2025-11-17T21:28:18.456Z",
        "thought": "Optimizing entry logic",
        "context": {"optimization": "limit_orders"}
      }
    ]
  },
  "total_iterations": 5,
  "total_activities": 248
}
```

---

## 2. Frontend React Components

### Dashboard Container Component
```javascript
import React, { useState, useEffect } from 'react';

const AgentDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Poll every 5 seconds for updates
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/api/agent-dashboard');
        if (!response.ok) throw new Error('Failed to fetch dashboard data');
        const data = await response.json();
        setDashboardData(data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }, []);

  if (loading) return <div>Loading agent dashboard...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!dashboardData) return <div>No data available</div>;

  return (
    <div className="agent-dashboard">
      <AgentStatusGrid agentStatus={dashboardData.agent_status} />
      <RecentActivitiesFeed activities={dashboardData.recent_activities} />
      <StrategyDetails strategy={dashboardData.latest_strategy} />
      <IterationSummary iteration={dashboardData.latest_iteration} />
      <AgentThoughts thoughts={dashboardData.agent_thoughts_summary} />
    </div>
  );
};

export default AgentDashboard;
```

### Agent Status Grid Component
```javascript
const AgentStatusGrid = ({ agentStatus }) => {
  return (
    <div className="agent-status-grid">
      <h2>Agent Status</h2>
      {Object.entries(agentStatus).map(([agentName, status]) => (
        <div key={agentName} className="agent-card">
          <h3>{agentName}</h3>
          <p className="last-active">
            Last Active: {new Date(status.last_active).toLocaleTimeString()}
          </p>
          <p className="last-thought">{status.last_thought}</p>
          <p className="activity-count">
            Total Activities: {status.activity_count}
          </p>
        </div>
      ))}
    </div>
  );
};
```

### Recent Activities Feed Component
```javascript
const RecentActivitiesFeed = ({ activities }) => {
  return (
    <div className="activities-feed">
      <h2>Recent Activities</h2>
      <div className="activities-list">
        {activities.map((activity, index) => (
          <div key={index} className={`activity-item ${activity.type}`}>
            <div className="activity-header">
              <span className="agent-name">{activity.agent}</span>
              <span className="timestamp">
                {new Date(activity.timestamp).toLocaleTimeString()}
              </span>
            </div>
            
            {activity.type === 'thinking' && (
              <div className="activity-content">
                <p>{activity.thought}</p>
                {activity.context && (
                  <pre className="context">
                    {JSON.stringify(activity.context, null, 2)}
                  </pre>
                )}
              </div>
            )}
            
            {activity.type === 'suggestion' && (
              <div className="activity-content">
                <p className="suggestion-type">{activity.suggestion_type}</p>
                <pre className="suggestion">
                  {JSON.stringify(activity.suggestion, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
```

### Strategy Details Component
```javascript
const StrategyDetails = ({ strategy }) => {
  if (!strategy || !strategy.strategy_data) {
    return <div>No strategy data available</div>;
  }

  const { strategy_data } = strategy;

  return (
    <div className="strategy-details">
      <h2>Latest Strategy (Iteration {strategy.iteration})</h2>
      
      <section className="alpha-sources">
        <h3>Alpha Sources Used</h3>
        {strategy_data.alpha_sources_used.map((source, idx) => (
          <div key={idx} className="alpha-source">
            <p><strong>{source.name}</strong></p>
            <p>Expected Sharpe: {source.expected_sharpe}</p>
            <p>Status: {source.implementation_status}</p>
          </div>
        ))}
      </section>

      <section className="optimizations">
        <h3>Execution Optimizations Applied</h3>
        <ul>
          {strategy_data.execution_optimizations_applied.map((opt, idx) => (
            <li key={idx}>{opt}</li>
          ))}
        </ul>
      </section>

      <section className="risk-params">
        <h3>Risk Parameters</h3>
        <pre>{JSON.stringify(strategy_data.risk_parameters, null, 2)}</pre>
      </section>

      <section className="expected-performance">
        <h3>Expected Performance</h3>
        <pre>{JSON.stringify(strategy_data.expected_performance, null, 2)}</pre>
      </section>

      <section className="generated-files">
        <h3>Generated Files</h3>
        <ul>
          {strategy_data.code_files_generated.map((file, idx) => (
            <li key={idx}>{file}</li>
          ))}
        </ul>
      </section>
    </div>
  );
};
```

### Iteration Summary Component
```javascript
const IterationSummary = ({ iteration }) => {
  if (!iteration || !iteration.summary) {
    return <div>No iteration data available</div>;
  }

  const { summary } = iteration;

  return (
    <div className="iteration-summary">
      <h2>Iteration {iteration.iteration} Summary</h2>
      
      <section className="research-inputs">
        <h3>Research Agent Inputs</h3>
        <ul>
          {summary.research_agent_inputs.map((input, idx) => (
            <li key={idx}>{input}</li>
          ))}
        </ul>
      </section>

      <section className="execution-opts">
        <h3>Execution Agent Optimizations</h3>
        <ul>
          {summary.execution_agent_optimizations.map((opt, idx) => (
            <li key={idx}>{opt}</li>
          ))}
        </ul>
      </section>

      <section className="risk-validations">
        <h3>Risk Agent Validations</h3>
        <ul>
          {summary.risk_agent_validations.map((val, idx) => (
            <li key={idx}>{val}</li>
          ))}
        </ul>
      </section>

      <section className="reasoner-decisions">
        <h3>Reasoner Decisions</h3>
        <ul>
          {summary.reasoner_decisions.map((decision, idx) => (
            <li key={idx}>{decision}</li>
          ))}
        </ul>
      </section>

      <section className="coder-outputs">
        <h3>Coder Outputs</h3>
        <ul>
          {summary.coder_outputs.map((output, idx) => (
            <li key={idx}>{output}</li>
          ))}
        </ul>
      </section>

      <section className="final-results">
        <h3>Final Results</h3>
        <div className="results-grid">
          <div>Status: <strong>{summary.final_results.status}</strong></div>
          <div>Alpha Score: <strong>{summary.final_results.alpha_score}</strong></div>
          <div>Total Trades: <strong>{summary.final_results.total_trades}</strong></div>
          <div>Diversity: <strong>{summary.final_results.diversity}</strong></div>
          <div>Avg Confidence: <strong>{summary.final_results.avg_confidence}</strong></div>
        </div>
      </section>
    </div>
  );
};
```

### Agent Thoughts Component
```javascript
const AgentThoughts = ({ thoughts }) => {
  return (
    <div className="agent-thoughts">
      <h2>Agent Thoughts Summary</h2>
      {Object.entries(thoughts).map(([agentName, thoughtList]) => (
        <div key={agentName} className="agent-thought-section">
          <h3>{agentName}</h3>
          {thoughtList.map((thought, idx) => (
            <div key={idx} className="thought-item">
              <p className="timestamp">
                {new Date(thought.timestamp).toLocaleString()}
              </p>
              <p className="thought">{thought.thought}</p>
              {thought.context && (
                <pre className="context">
                  {JSON.stringify(thought.context, null, 2)}
                </pre>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
};
```

---

## 3. Backend Flask API Implementation

```python
from flask import Flask, jsonify
from monolith import AgentActivityTracker

app = Flask(__name__)
activity_tracker = AgentActivityTracker("agent_activity.json")
activity_tracker.load_from_file()

@app.route('/api/agent-dashboard', methods=['GET'])
def get_agent_dashboard():
    """Return complete dashboard data for frontend"""
    dashboard_data = activity_tracker.get_frontend_dashboard_data()
    return jsonify(dashboard_data)

@app.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Return just agent status"""
    status = activity_tracker.get_all_agent_status()
    return jsonify(status)

@app.route('/api/recent-activities', methods=['GET'])
def get_recent_activities():
    """Return recent activities"""
    activities = activity_tracker.get_recent_activities(limit=50)
    return jsonify(activities)

@app.route('/api/strategy/<int:iteration>', methods=['GET'])
def get_strategy_details(iteration):
    """Return strategy details for specific iteration"""
    strategy = activity_tracker.get_strategy_details(iteration)
    return jsonify(strategy)

@app.route('/api/iteration/<int:iteration>', methods=['GET'])
def get_iteration_summary(iteration):
    """Return iteration summary"""
    summary = activity_tracker.get_iteration_summary(iteration)
    return jsonify(summary)

@app.route('/api/agent-thoughts/<agent_name>', methods=['GET'])
def get_agent_thoughts(agent_name):
    """Return thoughts from specific agent"""
    thoughts = activity_tracker.get_agent_thoughts(agent_name, limit=20)
    return jsonify(thoughts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

---

## 4. WebSocket Support for Real-Time Updates

### Backend WebSocket Server
```python
from flask_socketio import SocketIO, emit
from flask import Flask
from monolith import AgentActivityTracker
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
activity_tracker = AgentActivityTracker("agent_activity.json")
activity_tracker.load_from_file()

def broadcast_updates():
    """Background thread to broadcast updates"""
    while True:
        dashboard_data = activity_tracker.get_frontend_dashboard_data()
        socketio.emit('dashboard_update', dashboard_data)
        time.sleep(2)  # Update every 2 seconds

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send initial data
    dashboard_data = activity_tracker.get_frontend_dashboard_data()
    emit('dashboard_update', dashboard_data)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Start background update thread
    update_thread = threading.Thread(target=broadcast_updates)
    update_thread.daemon = True
    update_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
```

### Frontend WebSocket Client
```javascript
import io from 'socket.io-client';
import React, { useState, useEffect } from 'react';

const AgentDashboardLive = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const socket = io('http://localhost:5000');

    socket.on('connect', () => {
      console.log('Connected to backend');
      setConnected(true);
    });

    socket.on('dashboard_update', (data) => {
      console.log('Received dashboard update');
      setDashboardData(data);
    });

    socket.on('disconnect', () => {
      console.log('Disconnected from backend');
      setConnected(false);
    });

    return () => socket.disconnect();
  }, []);

  if (!connected) return <div>Connecting to backend...</div>;
  if (!dashboardData) return <div>Waiting for data...</div>;

  return (
    <div className="agent-dashboard-live">
      <div className="connection-status">
        Status: {connected ? 'Connected' : 'Disconnected'}
      </div>
      <AgentStatusGrid agentStatus={dashboardData.agent_status} />
      <RecentActivitiesFeed activities={dashboardData.recent_activities} />
      <StrategyDetails strategy={dashboardData.latest_strategy} />
      <IterationSummary iteration={dashboardData.latest_iteration} />
    </div>
  );
};

export default AgentDashboardLive;
```

---

## 5. Data Flow Summary

```
Backend (e17_final)
    ↓
MonolithAlphaEngine
    ↓
AgentActivityTracker.log_agent_thinking()
AgentActivityTracker.log_agent_suggestion()
AgentActivityTracker.log_strategy_info()
    ↓
agent_activity.json (persisted)
    ↓
Flask API (/api/agent-dashboard)
    ↓
JSON Response
    ↓
Frontend (React/Vue/Angular)
    ↓
Display: Agent Status, Activities, Strategies, Iterations
```

---

## 6. Complete Integration Example

```bash
# Backend setup
cd /path/to/monolith
python flask_api.py

# Frontend setup
cd /path/to/frontend
npm install socket.io-client
npm start

# Access dashboard
http://localhost:3000/dashboard
```

---

## 7. JSON Data Samples

### Agent Thinking Event
```json
{
  "timestamp": "2025-11-17T21:28:25.345Z",
  "unix_time": 1700255305.345,
  "agent": "ResearchQuant",
  "thought": "Analyzing volatility risk premium for current market regime",
  "context": {
    "market_regime": "high_volatility",
    "vix_level": 28.5,
    "recommended_strategy": "sell_vol"
  },
  "type": "thinking"
}
```

### Agent Suggestion Event
```json
{
  "timestamp": "2025-11-17T21:28:30.123Z",
  "unix_time": 1700255310.123,
  "agent": "ExecutionQuant",
  "suggestion_type": "entry_optimization",
  "suggestion": {
    "method": "limit_orders",
    "price_offset": "mid_price",
    "time_in_force": "GTC",
    "expected_slippage_reduction": 0.0025
  },
  "type": "suggestion"
}
```

### Strategy Info Event
```json
{
  "timestamp": "2025-11-17T21:28:00.000Z",
  "unix_time": 1700255280.0,
  "iteration": 5,
  "strategy_data": {
    "alpha_sources_used": ["volatility_risk_premium", "momentum_crash_protection"],
    "execution_optimizations_applied": ["limit_orders", "atr_stops"],
    "risk_parameters": {"max_dd": 0.15, "var_95": 0.02},
    "expected_performance": {"sharpe": 1.5, "max_dd": 0.12}
  },
  "type": "strategy_info"
}
```

---

**All code above is production-ready. Frontend developers can copy-paste and implement immediately.**
