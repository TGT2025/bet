# E17 Trading System - Frontend API Documentation

## 📊 Complete API Reference for Dashboard Development

This document provides comprehensive details on all monitoring endpoints, data structures, and recommended visualizations for building a real-time trading system dashboard.

---

## 🌐 API Base Configuration

```
Base URL: http://localhost:8000
Protocol: HTTP/REST + WebSocket
Format: JSON
CORS: Enabled (allow all origins in dev)
```

---

## 📍 REST API Endpoints

### 1. System Status
**Endpoint:** `GET /api/status`

**Description:** Real-time system health and metrics

**Response:**
```json
{
  "status": "running",
  "uptime_seconds": 3600,
  "iteration": 42,
  "champions": 3,
  "last_heartbeat": "2025-11-16T21:30:00Z",
  "health": "healthy",
  "version": "final"
}
```

**Recommended Chart:** 
- **Type:** Status Card + Gauge
- **Metrics:** Uptime (counter), Iteration (number), Champions (badge)
- **Colors:** Green (healthy), Yellow (warning), Red (critical)

---

### 2. Iteration History
**Endpoint:** `GET /api/iterations`

**Query Parameters:**
- `limit` (int): Number of iterations to return (default: 50)
- `offset` (int): Pagination offset (default: 0)

**Response:**
```json
{
  "iterations": [
    {
      "iteration": 42,
      "timestamp": "2025-11-16T21:25:00Z",
      "status": "champion_promoted",
      "alpha_score": 0.45,
      "performance": {
        "total_trades": 25,
        "win_rate": 0.60,
        "profit_factor": 1.35,
        "max_drawdown": 0.12,
        "sharpe_ratio": 1.15
      },
      "duration_seconds": 180,
      "champion_id": "champion_42_1700167500"
    }
  ],
  "total": 100,
  "page": 1
}
```

**Recommended Charts:**
1. **Timeline Chart** (Line Chart)
   - X-axis: Time
   - Y-axis: Alpha Score
   - Series: Alpha score progression
   - Color: Gradient (red → green as score increases)

2. **Success Rate** (Pie Chart)
   - Slices: Champion promoted, Completed (no champion), Failed
   - Colors: Green, Blue, Red

3. **Iteration Table** (Data Grid)
   - Columns: Iteration #, Status, Alpha Score, Win Rate, Trades, Duration
   - Sortable: Yes
   - Filters: Status, Date range

---

### 3. Champion Details
**Endpoint:** `GET /api/champions`

**Response:**
```json
{
  "champions": [
    {
      "id": "champion_42_1700167500",
      "iteration": 42,
      "created_at": "2025-11-16T21:25:00Z",
      "status": "active",
      "performance": {
        "total_trades": 25,
        "win_rate": 0.60,
        "profit_factor": 1.35,
        "max_drawdown": 0.12,
        "total_pnl": 1250.50,
        "avg_trade": 50.02
      },
      "current_equity": 11250.50,
      "trades_today": 12,
      "last_trade_at": "2025-11-16T21:30:00Z"
    }
  ],
  "total_champions": 3,
  "active_champions": 2
}
```

**Recommended Charts:**
1. **Champion Cards** (Card Grid)
   - Layout: 3-column grid
   - Content: ID, Win Rate badge, PnL (colored), Status
   - Actions: View details, Pause, Stop

2. **Performance Comparison** (Grouped Bar Chart)
   - X-axis: Champion IDs
   - Y-axis: Metrics (Win Rate, Profit Factor, Sharpe Ratio)
   - Grouped bars for each metric

3. **Equity Curve** (Area Chart)
   - X-axis: Time
   - Y-axis: Equity
   - Series: One per champion
   - Fill: Semi-transparent

---

### 4. Real-Time Trades
**Endpoint:** `GET /api/trades`

**Query Parameters:**
- `champion_id` (string): Filter by champion
- `limit` (int): Number of trades (default: 100)
- `since` (ISO timestamp): Trades after this time

**Response:**
```json
{
  "trades": [
    {
      "id": "trade_12345",
      "timestamp": "2025-11-16T21:29:55Z",
      "champion_id": "champion_42_1700167500",
      "symbol": "BTCUSDT",
      "action": "BUY",
      "price": 45123.50,
      "size": 0.05,
      "value": 2256.18,
      "strategy": "momentum_reversal",
      "confidence": 0.75,
      "pnl": null,
      "status": "open"
    },
    {
      "id": "trade_12344",
      "timestamp": "2025-11-16T21:28:00Z",
      "champion_id": "champion_42_1700167500",
      "symbol": "ETHUSDT",
      "action": "SELL",
      "price": 2345.60,
      "size": 1.5,
      "value": 3518.40,
      "strategy": "mean_reversion",
      "confidence": 0.82,
      "pnl": 125.50,
      "status": "closed"
    }
  ],
  "total": 1250
}
```

**Recommended Charts:**
1. **Trade Feed** (Live Table)
   - Columns: Time, Symbol, Action, Price, Size, PnL, Status
   - Auto-refresh: Every 5 seconds
   - Row colors: Green (profit), Red (loss), Blue (open)

2. **Trade Volume Over Time** (Bar Chart)
   - X-axis: Time (hourly buckets)
   - Y-axis: Number of trades
   - Series: Buys (green), Sells (red)

3. **Symbol Distribution** (Donut Chart)
   - Slices: Trading volume per symbol
   - Labels: Symbol + percentage

4. **PnL Timeline** (Candlestick Chart)
   - X-axis: Time
   - Y-axis: Cumulative PnL
   - Overlay: Trade markers

---

### 5. Agent Activity
**Endpoint:** `GET /api/agents`

**Response:**
```json
{
  "agents": [
    {
      "name": "whale_hunter_agent",
      "status": "active",
      "last_run": "2025-11-16T21:30:00Z",
      "suggestions_count": 15,
      "confidence_avg": 0.72,
      "performance_score": 0.65,
      "memory_loaded": true,
      "latest_suggestion": {
        "signal": "BUY",
        "symbol": "BTCUSDT",
        "confidence": 0.78,
        "reasoning": "Large accumulation pattern detected at $45000 support"
      }
    },
    {
      "name": "liquidity_miner_agent",
      "status": "active",
      "last_run": "2025-11-16T21:30:05Z",
      "suggestions_count": 12,
      "confidence_avg": 0.68,
      "performance_score": 0.58,
      "memory_loaded": true,
      "latest_suggestion": {
        "signal": "SELL",
        "symbol": "ETHUSDT",
        "confidence": 0.71,
        "reasoning": "Liquidity imbalance at resistance zone"
      }
    },
    {
      "name": "volatility_predictor_agent",
      "status": "active",
      "last_run": "2025-11-16T21:30:10Z",
      "suggestions_count": 10,
      "confidence_avg": 0.75,
      "performance_score": 0.70,
      "memory_loaded": true,
      "latest_suggestion": {
        "signal": "NEUTRAL",
        "symbol": "BTCUSDT",
        "confidence": 0.65,
        "reasoning": "Volatility compression expected, wait for breakout"
      }
    }
  ]
}
```

**Recommended Charts:**
1. **Agent Status Cards** (Card Grid)
   - Layout: 3 columns
   - Content: Agent name, status badge, performance score (progress bar)
   - Live update: Every 10 seconds

2. **Agent Performance** (Radar Chart)
   - Axes: Confidence, Accuracy, Suggestion Count, Performance Score
   - Series: One per agent
   - Fill: Semi-transparent with different colors

3. **Suggestion Timeline** (Stacked Area Chart)
   - X-axis: Time
   - Y-axis: Number of suggestions
   - Stacks: One per agent
   - Colors: Distinct per agent

4. **Agent Insights Feed** (Scrollable List)
   - Items: Latest suggestion from each agent
   - Format: Card with agent icon, signal badge, reasoning text
   - Update: Real-time

---

### 6. System Logs
**Endpoint:** `GET /api/logs`

**Query Parameters:**
- `level` (string): Filter by log level (INFO, WARNING, ERROR, CRITICAL)
- `component` (string): Filter by component (REASONER, CODER, ENFORCER, etc.)
- `limit` (int): Number of logs (default: 100)
- `since` (ISO timestamp): Logs after this time

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2025-11-16T21:30:15.123Z",
      "level": "INFO",
      "component": "ALPHA-HUNTER",
      "message": "🔄 CONTINUOUS-ALPHA Iteration 43",
      "iteration": 43
    },
    {
      "timestamp": "2025-11-16T21:30:10.456Z",
      "level": "INFO",
      "component": "CHAMPION",
      "message": "🏆 NEW_CHAMPION_PROMOTED: champion_42_1700167500",
      "iteration": 42,
      "champion_id": "champion_42_1700167500"
    },
    {
      "timestamp": "2025-11-16T21:29:55.789Z",
      "level": "WARNING",
      "component": "ENFORCER",
      "message": "🎯 ALPHA_SCORE_LOW: 0.25",
      "iteration": 41,
      "alpha_score": 0.25
    }
  ],
  "total": 5432
}
```

**Recommended Charts:**
1. **Log Stream** (Virtual Scrolling List)
   - Items: Log entries with timestamp, level badge, component, message
   - Colors by level: Blue (INFO), Yellow (WARNING), Red (ERROR)
   - Auto-scroll: Optional toggle
   - Search: Full-text search

2. **Log Level Distribution** (Horizontal Bar Chart)
   - Y-axis: Log levels
   - X-axis: Count
   - Colors: Blue, Yellow, Orange, Red

3. **Component Activity** (Heatmap)
   - X-axis: Time (15-min buckets)
   - Y-axis: Components
   - Color intensity: Log count in bucket

---

### 7. Performance Metrics
**Endpoint:** `GET /api/metrics`

**Query Parameters:**
- `period` (string): Time period (1h, 6h, 24h, 7d, 30d)

**Response:**
```json
{
  "period": "24h",
  "summary": {
    "total_trades": 245,
    "win_rate": 0.58,
    "total_pnl": 3450.25,
    "avg_trade_pnl": 14.08,
    "max_drawdown": 0.15,
    "sharpe_ratio": 1.25,
    "profit_factor": 1.42,
    "best_trade": 250.50,
    "worst_trade": -120.25
  },
  "by_champion": [
    {
      "champion_id": "champion_42_1700167500",
      "trades": 100,
      "win_rate": 0.62,
      "pnl": 1520.30
    }
  ],
  "by_symbol": [
    {
      "symbol": "BTCUSDT",
      "trades": 120,
      "win_rate": 0.60,
      "pnl": 1850.15
    }
  ],
  "by_strategy": [
    {
      "strategy": "momentum_reversal",
      "trades": 80,
      "win_rate": 0.65,
      "pnl": 1200.50
    }
  ]
}
```

**Recommended Charts:**
1. **Performance Dashboard** (KPI Cards Grid)
   - Cards: Total PnL, Win Rate, Sharpe Ratio, Drawdown
   - Layout: 2×2 grid
   - Value formatting: Currency, Percentage
   - Trend indicators: Up/down arrows with delta

2. **PnL Over Time** (Line Chart with Area Fill)
   - X-axis: Time
   - Y-axis: Cumulative PnL
   - Fill: Green above 0, Red below 0
   - Reference line: Break-even (y=0)

3. **Win Rate by Period** (Line Chart)
   - X-axis: Time periods
   - Y-axis: Win rate percentage
   - Target line: 50% (break-even)
   - Colors: Green above 50%, Red below

4. **Performance Breakdown** (Treemap)
   - Hierarchy: Champion → Strategy → Symbol
   - Size: Number of trades
   - Color: PnL (red → green gradient)

5. **Drawdown Chart** (Area Chart)
   - X-axis: Time
   - Y-axis: Drawdown percentage
   - Fill: Red (inverted area)
   - Annotation: Max drawdown point

---

### 8. Heartbeat Monitor
**Endpoint:** `GET /api/heartbeat`

**Response:**
```json
{
  "last_beat": "2025-11-16T21:30:00Z",
  "seconds_since_beat": 15,
  "status": "alive",
  "uptime_seconds": 7200,
  "stats": {
    "iterations": 43,
    "champions": 3,
    "errors": 2,
    "restarts": 0
  }
}
```

**Recommended Charts:**
1. **Heartbeat Indicator** (Pulsing Icon)
   - Visual: Heart icon that pulses on each beat
   - Color: Green (alive), Yellow (warning), Red (timeout)
   - Text: "Last beat: 15s ago"

2. **Uptime Counter** (Digital Display)
   - Format: "7d 12h 30m 15s" or "7,200 seconds"
   - Style: Large, prominent
   - Updates: Every second

---

### 9. Error Log
**Endpoint:** `GET /api/errors`

**Query Parameters:**
- `limit` (int): Number of errors (default: 50)
- `since` (ISO timestamp): Errors after this time

**Response:**
```json
{
  "errors": [
    {
      "timestamp": "2025-11-16T20:15:30Z",
      "iteration": 38,
      "error_type": "RuntimeError",
      "message": "Strategy validation failed",
      "traceback": "Traceback (most recent call last)...",
      "consecutive_failures": 1,
      "recovered": true,
      "recovery_time_seconds": 30
    }
  ],
  "total_errors": 15,
  "errors_last_hour": 2,
  "recovery_rate": 0.93
}
```

**Recommended Charts:**
1. **Error Timeline** (Scatter Plot)
   - X-axis: Time
   - Y-axis: Error types (categorical)
   - Point size: Severity
   - Color: Red (not recovered), Yellow (recovered)

2. **Error Rate** (Line Chart)
   - X-axis: Time (hourly)
   - Y-axis: Errors per hour
   - Threshold line: Acceptable error rate
   - Alert: When exceeded

3. **Recovery Stats** (Progress Bar)
   - Label: "Recovery Rate: 93%"
   - Color: Green (>90%), Yellow (70-90%), Red (<70%)

---

### 10. Checkpoints
**Endpoint:** `GET /api/checkpoints`

**Response:**
```json
{
  "checkpoints": [
    {
      "iteration": 42,
      "timestamp": "2025-11-16T21:25:00Z",
      "file": "checkpoint_iter42_20251116_212500.pkl",
      "size_bytes": 15420,
      "champions_count": 2
    },
    {
      "iteration": 41,
      "timestamp": "2025-11-16T21:20:00Z",
      "file": "checkpoint_iter41_20251116_212000.pkl",
      "size_bytes": 14850,
      "champions_count": 2
    }
  ],
  "latest_checkpoint": {
    "iteration": 42,
    "can_resume": true
  },
  "total_checkpoints": 10
}
```

**Recommended Charts:**
1. **Checkpoint List** (Table)
   - Columns: Iteration, Time, Size, Champions
   - Actions: Download, Restore, Delete
   - Highlight: Latest checkpoint

2. **Checkpoint Timeline** (Timeline Visualization)
   - Events: Checkpoint saves
   - Markers: Circle icons on timeline
   - Tooltip: Iteration, time, champions

---

## 🔌 WebSocket Endpoints

### Real-Time Updates
**Endpoint:** `ws://localhost:8000/ws`

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

**Message Types:**

#### 1. Iteration Update
```json
{
  "type": "iteration_update",
  "data": {
    "iteration": 43,
    "status": "in_progress",
    "phase": "expert_coder",
    "progress": 0.35
  }
}
```

#### 2. Trade Executed
```json
{
  "type": "trade_executed",
  "data": {
    "trade_id": "trade_12345",
    "champion_id": "champion_42_1700167500",
    "symbol": "BTCUSDT",
    "action": "BUY",
    "price": 45123.50,
    "timestamp": "2025-11-16T21:30:00Z"
  }
}
```

#### 3. Champion Promoted
```json
{
  "type": "champion_promoted",
  "data": {
    "champion_id": "champion_43_1700167800",
    "iteration": 43,
    "alpha_score": 0.52,
    "performance": {
      "win_rate": 0.63,
      "profit_factor": 1.45
    }
  }
}
```

#### 4. Agent Suggestion
```json
{
  "type": "agent_suggestion",
  "data": {
    "agent": "whale_hunter_agent",
    "signal": "BUY",
    "symbol": "BTCUSDT",
    "confidence": 0.78,
    "reasoning": "Large accumulation detected"
  }
}
```

#### 5. Error Occurred
```json
{
  "type": "error",
  "data": {
    "iteration": 43,
    "error_type": "ValidationError",
    "message": "Alpha score too low",
    "severity": "warning"
  }
}
```

#### 6. Heartbeat
```json
{
  "type": "heartbeat",
  "data": {
    "timestamp": "2025-11-16T21:30:00Z",
    "status": "alive",
    "iterations": 43,
    "champions": 3
  }
}
```

**Recommended Visualizations:**
- **Live Activity Feed:** Scrollable list of all WebSocket messages
- **Toast Notifications:** Pop-up alerts for important events (champion promoted, errors)
- **Real-time Counters:** Auto-updating numbers for iterations, trades, etc.

---

## 📊 Recommended Dashboard Layout

### Page 1: Overview Dashboard
```
┌─────────────────────────────────────────────────┐
│  Header: Logo | System Status | Uptime          │
├───────────┬─────────────┬─────────────┬─────────┤
│  Total    │  Win Rate   │  Champions  │  Errors │
│   PnL     │    58%      │      3      │    2    │
├───────────┴─────────────┴─────────────┴─────────┤
│  PnL Over Time (Line Chart - Full Width)        │
├───────────────────────────┬─────────────────────┤
│  Iteration Timeline       │  Champion Cards     │
│  (Line Chart)             │  (Grid 3x1)         │
├───────────────────────────┼─────────────────────┤
│  Trade Feed               │  Agent Status       │
│  (Live Table)             │  (Radar Chart)      │
└───────────────────────────┴─────────────────────┘
```

### Page 2: Trading Activity
```
┌─────────────────────────────────────────────────┐
│  Filters: Time Range | Symbol | Champion        │
├───────────────────────────┬─────────────────────┤
│  Trade Table (Full Width) │                     │
├───────────────────────────┼─────────────────────┤
│  Symbol Distribution      │  Strategy           │
│  (Donut Chart)            │  Performance        │
│                           │  (Bar Chart)        │
├───────────────────────────┴─────────────────────┤
│  Candlestick Chart with Trade Markers           │
└─────────────────────────────────────────────────┘
```

### Page 3: Agent Insights
```
┌─────────────────────────────────────────────────┐
│  Agent Performance Comparison (Radar Chart)     │
├─────────────────┬─────────────────┬─────────────┤
│  Whale Hunter   │  Liquidity      │  Volatility │
│  Card + Chart   │  Miner Card     │  Predictor  │
├─────────────────┴─────────────────┴─────────────┤
│  Agent Suggestion Timeline (Stacked Area)       │
├─────────────────────────────────────────────────┤
│  Live Suggestion Feed (Scrollable Cards)        │
└─────────────────────────────────────────────────┘
```

### Page 4: System Monitoring
```
┌─────────────────────────────────────────────────┐
│  Heartbeat | Uptime | Checkpoints | Errors      │
├───────────────────────────┬─────────────────────┤
│  Log Stream               │  Error Timeline     │
│  (Virtual Scroll)         │  (Scatter Plot)     │
├───────────────────────────┼─────────────────────┤
│  Component Activity       │  System Health      │
│  (Heatmap)                │  (Gauges)           │
└───────────────────────────┴─────────────────────┘
```

---

## 🎨 Chart Library Recommendations

### Recommended Libraries:
1. **Chart.js** - Simple, lightweight, great for basic charts
2. **Apache ECharts** - Feature-rich, great for complex visualizations
3. **Recharts** - React-based, clean API, good TypeScript support
4. **D3.js** - Ultimate flexibility, steeper learning curve
5. **TradingView Lightweight Charts** - Perfect for financial candlestick charts

### Chart Type Guide:
- **Line Charts:** Time series data (PnL, alpha scores, metrics over time)
- **Bar Charts:** Comparisons (performance by strategy, trades per symbol)
- **Pie/Donut Charts:** Proportions (symbol distribution, success rates)
- **Area Charts:** Cumulative values (equity curve, drawdown)
- **Scatter Plots:** Events (errors, trade execution points)
- **Heatmaps:** Multi-dimensional data (component activity over time)
- **Radar Charts:** Multi-metric comparison (agent performance)
- **Candlestick Charts:** Price action with trade markers
- **Gauge Charts:** Single metric with thresholds (health scores)
- **Treemap:** Hierarchical data (performance breakdown)

---

## 🔄 Real-Time Update Strategy

### Polling Intervals:
- **Status:** Every 5 seconds
- **Trades:** Every 10 seconds
- **Agents:** Every 15 seconds
- **Metrics:** Every 30 seconds
- **Logs:** Every 10 seconds (with pagination)

### WebSocket Usage:
- **Use for:** Critical updates (trades, champions, errors)
- **Fallback:** Polling if WebSocket disconnects
- **Reconnect:** Exponential backoff (1s, 2s, 4s, 8s, max 30s)

---

## 🔐 Security Considerations

### API Authentication:
```javascript
// Add API key to headers
headers: {
  'Authorization': 'Bearer YOUR_API_KEY',
  'Content-Type': 'application/json'
}
```

### Rate Limiting:
- 100 requests per minute per IP
- WebSocket: 1 connection per client
- Bulk endpoints: 10 requests per minute

---

## 🧪 Testing Endpoints

### Sample cURL Commands:
```bash
# Get system status
curl http://localhost:8000/api/status

# Get latest iterations
curl http://localhost:8000/api/iterations?limit=10

# Get champion details
curl http://localhost:8000/api/champions

# Get recent trades
curl http://localhost:8000/api/trades?limit=50

# Get logs with filter
curl "http://localhost:8000/api/logs?level=ERROR&limit=20"
```

---

## 📱 Mobile Responsiveness

### Breakpoints:
- **Desktop:** > 1200px (Full dashboard)
- **Tablet:** 768px - 1199px (2-column layout)
- **Mobile:** < 768px (Single column, stacked cards)

### Mobile Optimizations:
- Simplified charts (fewer data points)
- Collapsible sections
- Touch-friendly controls
- Swipeable cards
- Bottom navigation

---

## 🎯 Performance Optimization

### Data Caching:
- Cache static data (historical trades, old iterations)
- Invalidate on updates
- Use service workers for offline access

### Chart Performance:
- Limit data points (max 500 per chart)
- Use data aggregation for large datasets
- Lazy load charts (render on scroll)
- Debounce updates (batch multiple changes)

### Bundle Size:
- Use tree-shaking
- Code splitting by route
- Lazy load heavy components (charts)
- Compress assets

---

## 📚 Example Frontend Code

### React Component Example:
```javascript
import { useEffect, useState } from 'react';
import { Line } from 'recharts';

function PnLChart() {
  const [data, setData] = useState([]);

  useEffect(() => {
    // Fetch metrics
    fetch('http://localhost:8000/api/metrics?period=24h')
      .then(res => res.json())
      .then(data => setData(data.pnl_timeline));

    // Set up WebSocket for real-time updates
    const ws = new WebSocket('ws://localhost:8000/ws');
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'trade_executed') {
        // Update chart data
        updatePnL(msg.data);
      }
    };

    return () => ws.close();
  }, []);

  return (
    <Line
      data={data}
      xKey="timestamp"
      yKey="pnl"
      stroke="#00ff00"
    />
  );
}
```

---

## ✅ Implementation Checklist

- [ ] Set up API client with error handling
- [ ] Implement WebSocket connection with reconnect logic
- [ ] Create reusable chart components
- [ ] Build responsive layout with grid system
- [ ] Add loading states and skeleton screens
- [ ] Implement error boundaries
- [ ] Add toast notifications for events
- [ ] Create theme system (light/dark mode)
- [ ] Add data export functionality (CSV, JSON)
- [ ] Implement user preferences (chart types, refresh rates)
- [ ] Add keyboard shortcuts
- [ ] Create comprehensive error messages
- [ ] Implement offline mode with service workers
- [ ] Add print/export dashboard views
- [ ] Create onboarding tour

---

## 🚀 Getting Started

1. **Install dependencies:**
   ```bash
   npm install chart.js recharts axios socket.io-client
   ```

2. **Test API connection:**
   ```javascript
   fetch('http://localhost:8000/api/status')
     .then(res => res.json())
     .then(data => console.log('Connected:', data));
   ```

3. **Start building:**
   - Begin with Overview Dashboard
   - Add one chart type at a time
   - Test with real data from running system
   - Add interactivity (filters, drill-downs)
   - Polish with animations and transitions

---

**Last Updated:** 2025-11-16  
**Version:** Final (P0+P1+P2)  
**Maintained by:** E17 Trading System Team
