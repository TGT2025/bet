# 🎨 E17 Frontend Integration Guide

**Complete reference for connecting your frontend to the E17 Trading System API**

---

## 📡 Base Configuration

```javascript
// API Configuration
const API_BASE_URL = 'http://157.180.54.22:8000';  // Your server IP
const API_ENDPOINTS = {
  status: '/api/status',
  iterations: '/api/iterations',
  champions: '/api/champions',
  heartbeat: '/api/heartbeat',
  errors: '/api/errors',
  logs: '/api/logs',
  agents: '/api/agents',
  metrics: '/api/metrics',
  checkpoints: '/api/checkpoints'
};

// Polling intervals (milliseconds)
const POLL_INTERVALS = {
  status: 5000,      // Poll every 5 seconds
  logs: 2000,        // Poll every 2 seconds (real-time feel)
  errors: 10000,     // Poll every 10 seconds
  heartbeat: 5000,   // Poll every 5 seconds
  iterations: 5000,  // Poll every 5 seconds
  champions: 10000,  // Poll every 10 seconds
  agents: 5000,      // Poll every 5 seconds
  metrics: 10000,    // Poll every 10 seconds
  checkpoints: 30000 // Poll every 30 seconds
};
```

---

## 🔌 1. System Status Endpoint

**Endpoint:** `GET /api/status`

### JSON Response Schema:
```json
{
  "status": "running",           // "running" | "crashed" | "starting"
  "uptime_seconds": 2288,
  "iteration": 2,
  "champions": 0,
  "last_heartbeat": 1763332522.628036,
  "health": "healthy",           // "healthy" | "warning" | "critical"
  "version": "final",
  "total_errors": 15,
  "total_logs": 847
}
```

### Frontend Implementation:
```javascript
// React Component Example
import { useState, useEffect } from 'react';

function SystemStatus() {
  const [status, setStatus] = useState(null);
  
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/status`);
        const data = await response.json();
        setStatus(data);
      } catch (error) {
        console.error('Failed to fetch status:', error);
      }
    };
    
    // Initial fetch
    fetchStatus();
    
    // Poll every 5 seconds
    const interval = setInterval(fetchStatus, POLL_INTERVALS.status);
    
    return () => clearInterval(interval);
  }, []);
  
  if (!status) return <div>Loading...</div>;
  
  return (
    <div className="status-card">
      <div className="status-badge" data-status={status.health}>
        {status.status.toUpperCase()}
      </div>
      <div className="metrics">
        <div className="metric">
          <span className="label">Uptime:</span>
          <span className="value">{formatUptime(status.uptime_seconds)}</span>
        </div>
        <div className="metric">
          <span className="label">Iteration:</span>
          <span className="value">{status.iteration}</span>
        </div>
        <div className="metric">
          <span className="label">Champions:</span>
          <span className="value">{status.champions}</span>
        </div>
        <div className="metric">
          <span className="label">Total Errors:</span>
          <span className="value">{status.total_errors}</span>
        </div>
      </div>
    </div>
  );
}

// Helper function
function formatUptime(seconds) {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${days}d ${hours}h ${mins}m`;
}
```

### Display on Frontend:
- **Card:** Show status badge (green/yellow/red based on health)
- **Metrics:** Display uptime, iteration, champions, errors
- **Indicator:** Animate heartbeat pulse if `last_heartbeat` recent (<60s ago)

---

## 📊 2. Iterations Endpoint

**Endpoint:** `GET /api/iterations`

### JSON Response Schema:
```json
{
  "iterations": [
    {
      "iteration": 2,
      "start_time": "2025-11-16T23:24:17.374216",
      "completed": false,
      "results": {},
      "phases": {
        "reasoner": {"status": "completed", "duration_s": 45.2},
        "coder": {"status": "completed", "duration_s": 120.5},
        "validator": {"status": "completed", "duration_s": 15.3},
        "tester": {"status": "running", "duration_s": null},
        "paper_trading": {"status": "pending", "duration_s": null},
        "champion_eval": {"status": "pending", "duration_s": null}
      },
      "files_generated": 9,
      "lines_of_code": 2847,
      "alpha_score": 0.35,
      "performance": {
        "total_trades": 47,
        "win_rate": 0.585,
        "profit_factor": 1.31,
        "max_drawdown": 0.182,
        "sharpe_ratio": 1.24
      }
    },
    {
      "iteration": 1,
      "start_time": "2025-11-16T22:59:14.402045",
      "completed": true,
      "results": {"status": "startup_failed"},
      "phases": {
        "reasoner": {"status": "completed", "duration_s": 38.1},
        "coder": {"status": "completed", "duration_s": 95.3},
        "validator": {"status": "failed", "duration_s": 5.2}
      }
    }
  ],
  "total": 2
}
```

### Frontend Implementation:
```javascript
function IterationTimeline() {
  const [iterations, setIterations] = useState([]);
  
  useEffect(() => {
    const fetchIterations = async () => {
      const response = await fetch(`${API_BASE_URL}/api/iterations`);
      const data = await response.json();
      setIterations(data.iterations);
    };
    
    fetchIterations();
    const interval = setInterval(fetchIterations, POLL_INTERVALS.iterations);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="iteration-timeline">
      {iterations.map(iter => (
        <div key={iter.iteration} className="iteration-card">
          <h3>Iteration {iter.iteration}</h3>
          <div className="timestamp">{new Date(iter.start_time).toLocaleString()}</div>
          
          {/* Phase Progress */}
          <div className="phase-progress">
            {Object.entries(iter.phases || {}).map(([phase, data]) => (
              <div key={phase} className={`phase phase-${data.status}`}>
                <span className="phase-name">{phase}</span>
                <span className="phase-status">{data.status}</span>
                {data.duration_s && (
                  <span className="phase-duration">{data.duration_s.toFixed(1)}s</span>
                )}
              </div>
            ))}
          </div>
          
          {/* Performance Metrics (if available) */}
          {iter.performance && (
            <div className="performance-metrics">
              <div>Trades: {iter.performance.total_trades}</div>
              <div>Win Rate: {(iter.performance.win_rate * 100).toFixed(1)}%</div>
              <div>Profit Factor: {iter.performance.profit_factor.toFixed(2)}</div>
              <div>Max DD: {(iter.performance.max_drawdown * 100).toFixed(1)}%</div>
            </div>
          )}
          
          {/* Code Generation */}
          {iter.files_generated && (
            <div className="code-stats">
              📁 {iter.files_generated} files, {iter.lines_of_code} lines
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
```

### Display on Frontend:
- **Timeline:** Vertical timeline showing each iteration
- **Phase Progress Bar:** 6 phases with color coding (green=completed, yellow=running, gray=pending, red=failed)
- **Metrics Cards:** Show performance when available
- **Current Iteration:** Highlight with animation/border

---

## 🏆 3. Champions Endpoint

**Endpoint:** `GET /api/champions`

### JSON Response Schema:
```json
{
  "champions": [
    {
      "name": "champion_2_1731798320",
      "iteration": 2,
      "promoted_at": "2025-11-16T23:45:12",
      "project_dir": "/tmp/alpha_hunt_iter2_20251116_233000",
      "performance": {
        "total_trades": 47,
        "win_rate": 0.585,
        "profit_factor": 1.31,
        "max_drawdown": 0.182,
        "sharpe_ratio": 1.24
      },
      "alpha_score": 0.35,
      "risk_params": {
        "max_position_size": 0.10,
        "stop_loss": 0.02,
        "take_profit": 0.05
      }
    }
  ],
  "total_champions": 1
}
```

### Frontend Implementation:
```javascript
function ChampionsList() {
  const [champions, setChampions] = useState([]);
  
  useEffect(() => {
    const fetchChampions = async () => {
      const response = await fetch(`${API_BASE_URL}/api/champions`);
      const data = await response.json();
      setChampions(data.champions);
    };
    
    fetchChampions();
    const interval = setInterval(fetchChampions, POLL_INTERVALS.champions);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="champions-grid">
      {champions.map(champ => (
        <div key={champ.name} className="champion-card">
          <div className="champion-header">
            <h3>🏆 {champ.name}</h3>
            <span className="iteration-badge">Iteration {champ.iteration}</span>
          </div>
          
          <div className="champion-metrics">
            <div className="metric-row">
              <span>Win Rate:</span>
              <span className="value">{(champ.performance.win_rate * 100).toFixed(1)}%</span>
            </div>
            <div className="metric-row">
              <span>Profit Factor:</span>
              <span className="value">{champ.performance.profit_factor.toFixed(2)}</span>
            </div>
            <div className="metric-row">
              <span>Sharpe Ratio:</span>
              <span className="value">{champ.performance.sharpe_ratio.toFixed(2)}</span>
            </div>
            <div className="metric-row">
              <span>Max Drawdown:</span>
              <span className="value danger">{(champ.performance.max_drawdown * 100).toFixed(1)}%</span>
            </div>
            <div className="metric-row">
              <span>Alpha Score:</span>
              <span className="value">{champ.alpha_score.toFixed(2)}</span>
            </div>
          </div>
          
          <div className="promoted-time">
            Promoted: {new Date(champ.promoted_at).toLocaleString()}
          </div>
        </div>
      ))}
      
      {champions.length === 0 && (
        <div className="no-champions">
          No champions yet. System is hunting for alpha...
        </div>
      )}
    </div>
  );
}
```

### Display on Frontend:
- **Grid:** 2-3 columns of champion cards
- **Highlight:** Gold border and trophy emoji
- **Performance Radar Chart:** Visualize multi-dimensional performance
- **Timeline:** Show when promoted

---

## ❌ 4. Errors Endpoint

**Endpoint:** `GET /api/errors`

### JSON Response Schema:
```json
{
  "errors": [
    {
      "timestamp": "2025-11-16T23:35:00.123456",
      "iteration": 2,
      "error": "Backtester execution failed: 'dict' object has no attribute 'columns'",
      "traceback": "Traceback (most recent call last):\n  File \"/tmp/alpha.py\", line 342, in run_backtest\n    df.columns = ['timestamp', 'price']\nAttributeError: 'dict' object has no attribute 'columns'",
      "consecutive_failures": 1,
      "error_type": "AttributeError"
    },
    {
      "timestamp": "2025-11-16T23:20:36.789012",
      "iteration": 1,
      "error": "Runtime error attempt 1",
      "traceback": "Traceback (most recent call last):\n  ...",
      "consecutive_failures": 1,
      "error_type": "RuntimeError"
    }
  ],
  "total_errors": 15,
  "errors_last_hour": 2,
  "recovery_rate": 0.93
}
```

### Frontend Implementation:
```javascript
function ErrorTimeline() {
  const [errorData, setErrorData] = useState(null);
  const [expandedError, setExpandedError] = useState(null);
  
  useEffect(() => {
    const fetchErrors = async () => {
      const response = await fetch(`${API_BASE_URL}/api/errors`);
      const data = await response.json();
      setErrorData(data);
    };
    
    fetchErrors();
    const interval = setInterval(fetchErrors, POLL_INTERVALS.errors);
    return () => clearInterval(interval);
  }, []);
  
  if (!errorData) return <div>Loading errors...</div>;
  
  return (
    <div className="error-container">
      {/* Summary Stats */}
      <div className="error-stats">
        <div className="stat">
          <span className="label">Total Errors:</span>
          <span className="value">{errorData.total_errors}</span>
        </div>
        <div className="stat">
          <span className="label">Last Hour:</span>
          <span className="value">{errorData.errors_last_hour}</span>
        </div>
        <div className="stat">
          <span className="label">Recovery Rate:</span>
          <span className="value success">
            {(errorData.recovery_rate * 100).toFixed(0)}%
          </span>
        </div>
      </div>
      
      {/* Error Timeline */}
      <div className="error-timeline">
        {errorData.errors.map((err, idx) => (
          <div key={idx} className="error-item">
            <div className="error-header" onClick={() => setExpandedError(
              expandedError === idx ? null : idx
            )}>
              <span className="error-type">{err.error_type}</span>
              <span className="error-time">
                {new Date(err.timestamp).toLocaleTimeString()}
              </span>
              <span className="iteration-badge">Iter {err.iteration}</span>
              <button className="expand-btn">
                {expandedError === idx ? '−' : '+'}
              </button>
            </div>
            
            <div className="error-message">{err.error}</div>
            
            {expandedError === idx && (
              <div className="error-traceback">
                <pre>{err.traceback}</pre>
              </div>
            )}
            
            {err.consecutive_failures > 1 && (
              <div className="failure-count">
                ⚠️ Failed {err.consecutive_failures} times consecutively
              </div>
            )}
          </div>
        ))}
      </div>
      
      {errorData.errors.length === 0 && (
        <div className="no-errors">
          ✅ No errors! System running smoothly.
        </div>
      )}
    </div>
  );
}
```

### Display on Frontend:
- **Scatter Plot:** Error timeline with dots
- **Expandable Cards:** Click to see full traceback
- **Stats Dashboard:** Show total, last hour, recovery rate
- **Color Coding:** Red for errors, yellow for warnings

---

## 📜 5. Logs Endpoint

**Endpoint:** `GET /api/logs?level=INFO&component=CODER&limit=100&since=2025-11-16T22:00:00`

### Query Parameters:
- `level` (optional): ERROR, WARNING, INFO, DEBUG
- `component` (optional): REASONER, CODER, VALIDATOR, TESTER, etc.
- `limit` (optional): Max number of logs (default 1000)
- `since` (optional): ISO timestamp to filter from

### JSON Response Schema:
```json
{
  "logs": [
    {
      "timestamp": "2025-11-17T01:23:58.123456",
      "level": "INFO",
      "component": "REASONER",
      "message": "🧠 PHASE_START: Building execution plan",
      "iteration": 1
    },
    {
      "timestamp": "2025-11-17T01:23:58.234567",
      "level": "INFO",
      "component": "SYSTEM",
      "message": "┌─────────────────────────────────────────────────────────┐",
      "iteration": 1
    },
    {
      "timestamp": "2025-11-17T01:23:58.345678",
      "level": "INFO",
      "component": "SYSTEM",
      "message": "│ 🧠 PHASE 1: SUPER REASONER                              │",
      "iteration": 1
    },
    {
      "timestamp": "2025-11-17T01:24:45.123456",
      "level": "INFO",
      "component": "CODER",
      "message": "✅ Code generated: 9 files, 2,847 lines total",
      "iteration": 1
    },
    {
      "timestamp": "2025-11-17T01:24:45.234567",
      "level": "INFO",
      "component": "CODER",
      "message": "   • exchange_client.py: 342 lines",
      "iteration": 1
    },
    {
      "timestamp": "2025-11-17T01:24:45.345678",
      "level": "INFO",
      "component": "CODER",
      "message": "   • strategies.py: 487 lines",
      "iteration": 1
    },
    {
      "timestamp": "2025-11-17T01:25:10.123456",
      "level": "ERROR",
      "component": "TESTER",
      "message": "Runtime error attempt 1: Traceback (most recent call last):",
      "iteration": 1
    }
  ],
  "total": 1000,
  "filtered": 847
}
```

### Frontend Implementation:
```javascript
function LogViewer() {
  const [logs, setLogs] = useState([]);
  const [filters, setFilters] = useState({
    level: 'ALL',
    component: 'ALL',
    search: ''
  });
  const [autoScroll, setAutoScroll] = useState(true);
  const logContainerRef = useRef(null);
  
  useEffect(() => {
    const fetchLogs = async () => {
      let url = `${API_BASE_URL}/api/logs?limit=500`;
      if (filters.level !== 'ALL') url += `&level=${filters.level}`;
      if (filters.component !== 'ALL') url += `&component=${filters.component}`;
      
      const response = await fetch(url);
      const data = await response.json();
      setLogs(data.logs);
      
      // Auto-scroll to bottom if enabled
      if (autoScroll && logContainerRef.current) {
        logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
      }
    };
    
    fetchLogs();
    const interval = setInterval(fetchLogs, POLL_INTERVALS.logs);
    return () => clearInterval(interval);
  }, [filters, autoScroll]);
  
  const filteredLogs = logs.filter(log => 
    !filters.search || log.message.toLowerCase().includes(filters.search.toLowerCase())
  );
  
  return (
    <div className="log-viewer">
      {/* Filter Controls */}
      <div className="log-controls">
        <select value={filters.level} onChange={(e) => setFilters({...filters, level: e.target.value})}>
          <option value="ALL">All Levels</option>
          <option value="ERROR">Errors</option>
          <option value="WARNING">Warnings</option>
          <option value="INFO">Info</option>
          <option value="DEBUG">Debug</option>
        </select>
        
        <select value={filters.component} onChange={(e) => setFilters({...filters, component: e.target.value})}>
          <option value="ALL">All Components</option>
          <option value="REASONER">Reasoner</option>
          <option value="CODER">Coder</option>
          <option value="VALIDATOR">Validator</option>
          <option value="TESTER">Tester</option>
          <option value="SYSTEM">System</option>
        </select>
        
        <input
          type="text"
          placeholder="Search logs..."
          value={filters.search}
          onChange={(e) => setFilters({...filters, search: e.target.value})}
        />
        
        <label>
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
          />
          Auto-scroll
        </label>
      </div>
      
      {/* Log Display */}
      <div className="log-container" ref={logContainerRef}>
        {filteredLogs.map((log, idx) => (
          <div 
            key={idx} 
            className={`log-entry log-${log.level.toLowerCase()}`}
          >
            <span className="log-timestamp">
              {new Date(log.timestamp).toLocaleTimeString()}
            </span>
            <span className="log-level">{log.level}</span>
            <span className="log-component">{log.component}</span>
            <span className="log-message">{log.message}</span>
            {log.iteration && (
              <span className="log-iteration">Iter {log.iteration}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Display on Frontend:
- **Terminal-style Viewer:** Monospace font, dark background
- **Color Coding:** ERROR=red, WARNING=yellow, INFO=white, DEBUG=gray
- **Auto-scroll:** Latest logs at bottom
- **Filter Bar:** Dropdowns for level, component, search
- **Highlight:** Phase banners (lines with ┌─ │ └─) in blue/cyan
- **Export:** Button to download logs as .txt file

---

## 💓 6. Heartbeat Endpoint

**Endpoint:** `GET /api/heartbeat`

### JSON Response Schema:
```json
{
  "status": "alive",
  "timestamp": 1763332522.628036,
  "uptime_seconds": 2288,
  "iteration": 2,
  "champions": 0,
  "total_errors": 15,
  "last_activity": "2025-11-17T01:25:22"
}
```

### Frontend Implementation:
```javascript
function HeartbeatMonitor() {
  const [heartbeat, setHeartbeat] = useState(null);
  const [isAlive, setIsAlive] = useState(true);
  
  useEffect(() => {
    const fetchHeartbeat = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/heartbeat`);
        const data = await response.json();
        setHeartbeat(data);
        
        // Check if heartbeat is recent (within last 60s)
        const age = Date.now() / 1000 - data.timestamp;
        setIsAlive(age < 60);
      } catch (error) {
        setIsAlive(false);
      }
    };
    
    fetchHeartbeat();
    const interval = setInterval(fetchHeartbeat, POLL_INTERVALS.heartbeat);
    return () => clearInterval(interval);
  }, []);
  
  if (!heartbeat) return <div>Waiting for heartbeat...</div>;
  
  const age = Date.now() / 1000 - heartbeat.timestamp;
  
  return (
    <div className={`heartbeat-monitor ${isAlive ? 'alive' : 'dead'}`}>
      <div className={`heartbeat-icon ${isAlive ? 'pulse' : ''}`}>
        {isAlive ? '💗' : '💔'}
      </div>
      <div className="heartbeat-info">
        <div className="status">{heartbeat.status.toUpperCase()}</div>
        <div className="last-beat">
          Last heartbeat: {age < 60 ? `${age.toFixed(0)}s ago` : 'TIMEOUT'}
        </div>
        <div className="beat-time">
          {new Date(heartbeat.last_activity).toLocaleString()}
        </div>
      </div>
    </div>
  );
}
```

### Display on Frontend:
- **Animated Heart:** Pulse animation if alive (<60s)
- **Status Badge:** Green/red based on heartbeat age
- **Timestamp:** Show last beat time
- **Alert:** Show warning if >60s since last beat

---

## 🤖 7. Agents Endpoint

**Endpoint:** `GET /api/agents`

### JSON Response Schema:
```json
{
  "agents": [
    {
      "timestamp": "2025-11-17T01:24:15.123456",
      "iteration": 1,
      "agent": "whale_hunter_agent",
      "action": "suggestion_made",
      "details": {
        "suggestion": "Detected whale accumulation in BTC/USDT",
        "confidence": 0.87,
        "signal_strength": "strong"
      }
    },
    {
      "timestamp": "2025-11-17T01:24:20.234567",
      "iteration": 1,
      "agent": "pattern_detector_agent",
      "action": "pattern_found",
      "details": {
        "pattern": "bullish_divergence",
        "timeframe": "4h",
        "symbol": "ETH/USDT"
      }
    }
  ],
  "total_activities": 150
}
```

### Frontend Implementation:
```javascript
function AgentActivity() {
  const [agents, setAgents] = useState([]);
  
  useEffect(() => {
    const fetchAgents = async () => {
      const response = await fetch(`${API_BASE_URL}/api/agents`);
      const data = await response.json();
      setAgents(data.agents);
    };
    
    fetchAgents();
    const interval = setInterval(fetchAgents, POLL_INTERVALS.agents);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="agent-activity">
      <h3>Agent Activity ({agents.length})</h3>
      <div className="agent-feed">
        {agents.map((activity, idx) => (
          <div key={idx} className="agent-card">
            <div className="agent-header">
              <span className="agent-name">🤖 {activity.agent}</span>
              <span className="agent-time">
                {new Date(activity.timestamp).toLocaleTimeString()}
              </span>
            </div>
            <div className="agent-action">{activity.action}</div>
            <div className="agent-details">
              {JSON.stringify(activity.details, null, 2)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Display on Frontend:
- **Feed:** Scrollable list of agent activities
- **Cards:** Each agent action as a card
- **Details:** Expandable JSON details
- **Badges:** Different colors per agent type

---

## 📈 8. Metrics Endpoint

**Endpoint:** `GET /api/metrics`

### JSON Response Schema:
```json
{
  "performance_metrics": {
    "total_trades": 47,
    "winning_trades": 28,
    "losing_trades": 19,
    "win_rate": 0.585,
    "profit_factor": 1.31,
    "sharpe_ratio": 1.24,
    "max_drawdown": 0.182,
    "total_pnl": 1250.75
  },
  "strategy_threshold_memory": {
    "min_trades": 20,
    "min_win_rate": 0.55,
    "min_profit_factor": 1.2,
    "max_drawdown": 0.3
  },
  "current_iteration": 2,
  "champions": 0
}
```

### Frontend Implementation:
```javascript
function PerformanceMetrics() {
  const [metrics, setMetrics] = useState(null);
  
  useEffect(() => {
    const fetchMetrics = async () => {
      const response = await fetch(`${API_BASE_URL}/api/metrics`);
      const data = await response.json();
      setMetrics(data);
    };
    
    fetchMetrics();
    const interval = setInterval(fetchMetrics, POLL_INTERVALS.metrics);
    return () => clearInterval(interval);
  }, []);
  
  if (!metrics) return <div>Loading metrics...</div>;
  
  const perf = metrics.performance_metrics;
  const thresh = metrics.strategy_threshold_memory;
  
  return (
    <div className="performance-dashboard">
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Total Trades</div>
          <div className="metric-value">{perf.total_trades}</div>
          <div className="metric-threshold">Min: {thresh.min_trades}</div>
        </div>
        
        <div className="metric-card">
          <div className="metric-label">Win Rate</div>
          <div className="metric-value">{(perf.win_rate * 100).toFixed(1)}%</div>
          <div className="metric-threshold">Min: {(thresh.min_win_rate * 100).toFixed(0)}%</div>
        </div>
        
        <div className="metric-card">
          <div className="metric-label">Profit Factor</div>
          <div className="metric-value">{perf.profit_factor.toFixed(2)}</div>
          <div className="metric-threshold">Min: {thresh.min_profit_factor}</div>
        </div>
        
        <div className="metric-card">
          <div className="metric-label">Max Drawdown</div>
          <div className="metric-value danger">{(perf.max_drawdown * 100).toFixed(1)}%</div>
          <div className="metric-threshold">Max: {(thresh.max_drawdown * 100).toFixed(0)}%</div>
        </div>
        
        <div className="metric-card">
          <div className="metric-label">Sharpe Ratio</div>
          <div className="metric-value">{perf.sharpe_ratio.toFixed(2)}</div>
        </div>
        
        <div className="metric-card">
          <div className="metric-label">Total PnL</div>
          <div className={`metric-value ${perf.total_pnl >= 0 ? 'profit' : 'loss'}`}>
            ${perf.total_pnl.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}
```

### Display on Frontend:
- **Grid:** 2x3 grid of metric cards
- **Progress Bars:** Show current vs threshold
- **Color Coding:** Green if above threshold, red if below
- **Charts:** Line chart for PnL over time

---

## 💾 9. Checkpoints Endpoint

**Endpoint:** `GET /api/checkpoints`

### JSON Response Schema:
```json
{
  "checkpoints": [
    {
      "file": "checkpoint_iter2_20251117_012500.pkl",
      "size_bytes": 15420,
      "modified": "2025-11-17T01:25:00",
      "iteration": 2
    },
    {
      "file": "checkpoint_iter1_20251117_010000.pkl",
      "size_bytes": 12830,
      "modified": "2025-11-17T01:00:00",
      "iteration": 1
    }
  ],
  "total_checkpoints": 2
}
```

### Frontend Implementation:
```javascript
function CheckpointManager() {
  const [checkpoints, setCheckpoints] = useState([]);
  
  useEffect(() => {
    const fetchCheckpoints = async () => {
      const response = await fetch(`${API_BASE_URL}/api/checkpoints`);
      const data = await response.json();
      setCheckpoints(data.checkpoints);
    };
    
    fetchCheckpoints();
    const interval = setInterval(fetchCheckpoints, POLL_INTERVALS.checkpoints);
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="checkpoint-manager">
      <h3>Checkpoints ({checkpoints.length})</h3>
      <div className="checkpoint-list">
        {checkpoints.map((cp, idx) => (
          <div key={idx} className="checkpoint-item">
            <div className="checkpoint-icon">💾</div>
            <div className="checkpoint-info">
              <div className="checkpoint-name">{cp.file}</div>
              <div className="checkpoint-meta">
                <span>Iteration {cp.iteration}</span>
                <span>{(cp.size_bytes / 1024).toFixed(1)} KB</span>
                <span>{new Date(cp.modified).toLocaleString()}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Display on Frontend:
- **List:** Show all checkpoints
- **Metadata:** File size, timestamp, iteration
- **Latest:** Highlight most recent checkpoint

---

## 🎯 Complete Dashboard Layout

```javascript
function E17Dashboard() {
  return (
    <div className="e17-dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <h1>E17 Trading System</h1>
        <SystemStatus />
        <HeartbeatMonitor />
      </header>
      
      {/* Main Content - 3 Column Layout */}
      <div className="dashboard-content">
        
        {/* Left Column - Iterations & Champions */}
        <div className="left-column">
          <section className="section">
            <h2>Iteration Progress</h2>
            <IterationTimeline />
          </section>
          
          <section className="section">
            <h2>Champions</h2>
            <ChampionsList />
          </section>
        </div>
        
        {/* Center Column - Logs & Errors */}
        <div className="center-column">
          <section className="section logs-section">
            <h2>System Logs</h2>
            <LogViewer />
          </section>
        </div>
        
        {/* Right Column - Metrics & Activity */}
        <div className="right-column">
          <section className="section">
            <h2>Performance Metrics</h2>
            <PerformanceMetrics />
          </section>
          
          <section className="section">
            <h2>Errors</h2>
            <ErrorTimeline />
          </section>
          
          <section className="section">
            <h2>Agent Activity</h2>
            <AgentActivity />
          </section>
          
          <section className="section">
            <h2>Checkpoints</h2>
            <CheckpointManager />
          </section>
        </div>
      </div>
    </div>
  );
}
```

---

## 🎨 CSS Styling Guide

```css
/* Dashboard Layout */
.e17-dashboard {
  font-family: 'Inter', system-ui, sans-serif;
  background: #0f172a;
  color: #e2e8f0;
  min-height: 100vh;
}

.dashboard-content {
  display: grid;
  grid-template-columns: 350px 1fr 350px;
  gap: 20px;
  padding: 20px;
}

/* Status Badge Colors */
.status-badge[data-status="healthy"] {
  background: #10b981;
  color: white;
}

.status-badge[data-status="warning"] {
  background: #f59e0b;
  color: white;
}

.status-badge[data-status="critical"] {
  background: #ef4444;
  color: white;
}

/* Log Entry Colors */
.log-entry.log-error {
  color: #ef4444;
  background: rgba(239, 68, 68, 0.1);
}

.log-entry.log-warning {
  color: #f59e0b;
}

.log-entry.log-info {
  color: #e2e8f0;
}

/* Heartbeat Animation */
@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.2); }
}

.heartbeat-icon.pulse {
  animation: pulse 1s infinite;
}

/* Champion Card */
.champion-card {
  border: 2px solid #fbbf24;
  background: linear-gradient(135deg, #78350f, #1e1b4b);
  border-radius: 8px;
  padding: 16px;
}

/* Metric Cards */
.metric-card {
  background: #1e293b;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}

.metric-value {
  font-size: 2rem;
  font-weight: bold;
}

.metric-value.profit { color: #10b981; }
.metric-value.loss { color: #ef4444; }
.metric-value.danger { color: #ef4444; }
```

---

## 📝 How to Get Logs to Pass for Checking

### Method 1: Copy from Frontend Log Viewer
1. Open your frontend dashboard
2. Go to the Logs section
3. Select filters: Level=ALL, Component=ALL
4. Copy all log entries
5. Save to file: `system_logs_for_review.txt`

### Method 2: Direct API Call
```bash
# Get all logs
curl http://157.180.54.22:8000/api/logs?limit=1000 > logs.json

# Get only errors
curl "http://157.180.54.22:8000/api/logs?level=ERROR&limit=500" > errors.json

# Get logs since specific time
curl "http://157.180.54.22:8000/api/logs?since=2025-11-17T01:00:00" > recent_logs.json

# Get coder phase logs
curl "http://157.180.54.22:8000/api/logs?component=CODER" > coder_logs.json
```

### Method 3: Export from Log Viewer Component
```javascript
// Add export button to LogViewer
function exportLogs() {
  const logText = logs.map(log => 
    `[${log.timestamp}] [${log.level}] [${log.component}] ${log.message}`
  ).join('\n');
  
  const blob = new Blob([logText], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `e17_logs_${new Date().toISOString()}.txt`;
  a.click();
}
```

### Method 4: Python Script to Fetch Logs
```python
import requests
import json
from datetime import datetime

# Fetch all logs
response = requests.get('http://157.180.54.22:8000/api/logs?limit=1000')
logs = response.json()['logs']

# Format and save
with open('system_logs_for_review.txt', 'w') as f:
    f.write(f"=== E17 SYSTEM LOGS ===\n")
    f.write(f"Generated: {datetime.now()}\n\n")
    
    for log in logs:
        timestamp = log['timestamp']
        level = log['level']
        component = log['component']
        message = log['message']
        iteration = log.get('iteration', 'N/A')
        
        f.write(f"[{timestamp}] [{level:8s}] [{component:10s}] [Iter {iteration}] {message}\n")

print("Logs saved to system_logs_for_review.txt")
```

---

## 🚀 Quick Start Checklist

- [ ] 1. Set `API_BASE_URL` to your server IP (157.180.54.22)
- [ ] 2. Test connection: `curl http://157.180.54.22:8000/api/status`
- [ ] 3. Implement `SystemStatus` component
- [ ] 4. Implement `IterationTimeline` component
- [ ] 5. Implement `LogViewer` component with filters
- [ ] 6. Implement `ErrorTimeline` component
- [ ] 7. Implement `ChampionsList` component
- [ ] 8. Implement `HeartbeatMonitor` component
- [ ] 9. Add polling intervals for real-time updates
- [ ] 10. Style with dark theme + color coding
- [ ] 11. Test all 9 endpoints
- [ ] 12. Add export functionality for logs/errors

---

## 📊 Data Flow Summary

```
E17 System (Backend)
    ↓
Flask API (0.0.0.0:8000)
    ↓
9 REST Endpoints
    ↓
Frontend (Poll every 2-30s)
    ↓
React Components
    ↓
User Dashboard
```

**All endpoints are LIVE and working!** 🎉

Start building your frontend with these exact JSON schemas and you'll have complete visibility into the E17 system.
