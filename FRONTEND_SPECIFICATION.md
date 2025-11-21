# APEX Frontend Specification & Dashboard Guide

## Complete Frontend Integration Documentation for APEX Trading System

**Version:** 1.0  
**System:** APEX (Autonomous Profit EXtraction)  
**Backend:** FastAPI on `http://localhost:8000`  
**Architecture:** Real-time WebSocket + REST API

---

## Table of Contents

1. [System Overview](#system-overview)
2. [API Endpoints Reference](#api-endpoints-reference)
3. [Frontend Dashboard Sections](#frontend-dashboard-sections)
4. [Real-Time Data Streaming](#real-time-data-streaming)
5. [Component Specifications](#component-specifications)
6. [Data Models & Structures](#data-models--structures)
7. [UI/UX Guidelines](#uiux-guidelines)

---

## System Overview

APEX is a fully autonomous trading system with **8 intelligent agents** working in synergy:

### Core Threads (5)
1. **Strategy Discovery Agent** - Finds trading strategies from web
2. **RBI Backtest Engine** - Tests and validates strategies
3. **Champion Manager** - Manages trading bots with 3-tier progression
4. **Market Data Agents** - Monitors whale/sentiment/funding
5. **API Server** - Provides real-time data to frontend

### Enhanced Intelligence Agents (3)
6. **Risk Agent** - Portfolio risk management & dynamic sizing
7. **Research Agent** - Strategy validation & quality analysis
8. **Volume Agent** - Volume confirmation & institutional flow detection

---

## API Endpoints Reference

### Base URL
```
http://localhost:8000
```

### REST Endpoints

#### 1. Root Endpoint
```http
GET /
```

**Response:**
```json
{
  "service": "APEX Autonomous Trading System",
  "version": "1.0",
  "status": "running",
  "threads": ["discovery", "rbi", "champion", "market_data", "api_server"],
  "uptime_seconds": 3600,
  "timestamp": "2025-11-21T18:00:00Z"
}
```

**Display:** System status banner at top of dashboard

---

#### 2. System Status
```http
GET /api/system_status
```

**Response:**
```json
{
  "threads": {
    "strategy_discovery": "RUNNING",
    "rbi_backtest": "RUNNING",
    "champion_manager": "RUNNING",
    "market_data": "RUNNING",
    "api_server": "RUNNING",
    "risk_agent": "RUNNING",
    "research_agent": "RUNNING",
    "volume_agent": "RUNNING"
  },
  "queues": {
    "strategy_discovery_queue": 2,
    "validated_strategy_queue": 0,
    "market_data_queue": 5
  },
  "system_health": {
    "cpu_usage": 45.2,
    "memory_usage": 68.5,
    "uptime_seconds": 86400,
    "last_error": null
  },
  "uptime_seconds": 86400,
  "last_errors": []
}
```

**Display:**
- Thread status indicators (green = RUNNING, red = STOPPED)
- Queue depths with progress bars
- System health metrics (CPU, memory, uptime)

---

#### 3. Champions List
```http
GET /api/champions
```

**Response:**
```json
{
  "champions": [
    {
      "id": "champion_1732211833",
      "status": "QUALIFIED",
      "bankroll": 11250.50,
      "profit_pct": 12.51,
      "total_trades": 75,
      "trades_today": 8,
      "winning_days": 5,
      "total_days": 7,
      "win_rate_days": 71.4,
      "win_rate_trades": 64.0,
      "profit_factor": 1.85,
      "sharpe_ratio": 1.42,
      "max_drawdown": 8.5,
      "current_positions": 2,
      "strategy_name": "Neural Network Volatility Forecasting",
      "real_trading_eligible": false,
      "created_at": "2025-11-19T10:00:00Z",
      "last_trade_at": "2025-11-21T17:45:00Z"
    }
  ],
  "summary": {
    "total_champions": 12,
    "elite": 1,
    "qualified": 3,
    "champions": 8,
    "total_bankroll": 125000.0,
    "total_profit": 8500.50,
    "overall_profit_pct": 6.8,
    "total_trades_today": 45
  }
}
```

**Display:**
- Champion cards grid with key metrics
- Status badges (CHAMPION/QUALIFIED/ELITE)
- Profit/loss indicators with color coding
- Mini charts for each champion's performance

---

#### 4. Strategy Discovery Status
```http
GET /api/strategy_discovery
```

**Response:**
```json
{
  "current_cycle": 5,
  "cycle_in_progress": false,
  "next_cycle_in": 1200,
  "recent_discoveries": [
    {
      "strategy_name": "Machine Learning Volatility Trading",
      "discovered_at": "2025-11-21T17:23:50Z",
      "source": "tavily",
      "status": "pending_backtest",
      "quality_score": "high"
    }
  ],
  "stats": {
    "total_discovered": 156,
    "pending_backtest": 3,
    "in_backtest": 1,
    "backtested": 153,
    "approved": 42,
    "rejected": 111,
    "approval_rate": 27.5
  }
}
```

**Display:**
- Discovery pipeline visualization
- Recent strategies table with status
- Stats cards for key metrics
- Countdown timer to next cycle

---

#### 5. Backtest Results
```http
GET /api/backtest_results
```

**Response:**
```json
{
  "recent_backtests": [
    {
      "strategy_name": "Neural Network Volatility Forecasting",
      "status": "APPROVED",
      "win_rate": 68.0,
      "profit_factor": 1.9,
      "sharpe_ratio": 1.4,
      "max_drawdown": 12.0,
      "total_trades": 118,
      "return_pct": 24.3,
      "llm_consensus": "APPROVED",
      "votes": {
        "deepseek": "APPROVE",
        "gpt4": "APPROVE",
        "claude": "REJECT"
      },
      "tested_configs": 6,
      "best_config": {
        "asset": "BTC",
        "timeframe": "15m",
        "period_days": 60
      },
      "tested_at": "2025-11-21T18:28:08Z"
    }
  ],
  "stats": {
    "total_backtests": 153,
    "approved": 42,
    "rejected": 111,
    "in_progress": 1,
    "avg_backtest_time": 285
  }
}
```

**Display:**
- Backtest results table with metrics
- LLM voting visualization (3 models)
- Performance charts for approved strategies
- Configuration comparison matrix

---

#### 6. Market Signals
```http
GET /api/market_signals
```

**Response:**
```json
{
  "whale_signals": [
    {
      "symbol": "btcusdt",
      "signal_type": "LARGE_OI_CHANGE",
      "change_pct": 124.76,
      "oi_value": "$3.13B",
      "action": "BUY",
      "confidence": 0.75,
      "detected_at": "2025-11-21T18:24:33Z"
    }
  ],
  "sentiment_signals": [
    {
      "sentiment": "BEARISH",
      "score": -0.86,
      "magnitude": "EXTREME",
      "sources": ["twitter", "reddit"],
      "confidence": 0.82,
      "detected_at": "2025-11-21T18:17:33Z"
    }
  ],
  "funding_signals": [
    {
      "symbol": "ethusdt",
      "rate": 0.0042,
      "action": "SHORT",
      "opportunity": "HIGH_FUNDING",
      "confidence": 0.68,
      "detected_at": "2025-11-21T18:15:00Z"
    }
  ],
  "volume_signals": [
    {
      "symbol": "btcusdt",
      "signal_type": "ACCUMULATION",
      "strength": 0.78,
      "price_level": 42150.50,
      "volume_profile": "HIGH",
      "detected_at": "2025-11-21T18:20:00Z"
    }
  ]
}
```

**Display:**
- Real-time signal feed with timestamps
- Signal type indicators with icons
- Confidence meters for each signal
- Action recommendations with color coding

---

#### 7. Risk Analytics
```http
GET /api/risk_analytics
```

**Response:**
```json
{
  "portfolio_risk": {
    "total_exposure": 35000.0,
    "max_exposure_limit": 45000.0,
    "utilization_pct": 77.8,
    "current_drawdown": 3.2,
    "max_drawdown_limit": 20.0,
    "var_95": 2150.50,
    "sharpe_ratio": 1.65,
    "sortino_ratio": 2.12,
    "calmar_ratio": 1.89
  },
  "position_correlation": [
    {
      "pair": ["BTC", "ETH"],
      "correlation": 0.85,
      "risk_level": "HIGH"
    }
  ],
  "risk_alerts": [
    {
      "type": "HIGH_CORRELATION",
      "message": "BTC and ETH positions highly correlated (0.85)",
      "severity": "MEDIUM",
      "recommendation": "Consider diversification"
    }
  ]
}
```

**Display:**
- Risk exposure gauge
- Correlation matrix heatmap
- Risk alerts with severity badges
- Performance ratio cards

---

#### 8. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-21T18:00:00Z"
}
```

**Display:** Health indicator dot (green/red)

---

## Frontend Dashboard Sections

### 1. Main Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│  APEX Trading System               🟢 RUNNING   Uptime: 24h │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐ │
│  │ Champions    │ Today's P/L  │ Total Trades │ Win Rate  │ │
│  │    12        │   +$850.50   │     45       │   64.2%   │ │
│  └──────────────┴──────────────┴──────────────┴───────────┘ │
│                                                               │
│  ┌───────────────────────────── CHAMPIONS ──────────────────┐│
│  │                                                           ││
│  │  [ELITE]   Neural Network Vol (ID:...833)                ││
│  │  ├ Bankroll: $11,250.50 (+12.5%)                         ││
│  │  ├ Win Rate: 64% │ Sharpe: 1.42 │ Trades: 75            ││
│  │  └ [View Details] [Performance Chart]                    ││
│  │                                                           ││
│  │  [QUALIFIED] Fibonacci Breakout (ID:...901)              ││
│  │  ├ Bankroll: $10,450.00 (+4.5%)                          ││
│  │  ├ Win Rate: 58% │ Sharpe: 1.15 │ Trades: 62            ││
│  │  └ [View Details] [Performance Chart]                    ││
│  │                                                           ││
│  └───────────────────────────────────────────────────────────┘│
│                                                               │
│  ┌──────────────────── REAL-TIME SIGNALS ────────────────────┐│
│  │  🐋 WHALE: BTC OI +124.76% → $3.13B (18:24:33)          ││
│  │  📊 SENTIMENT: EXTREME BEARISH -0.86 (18:17:33)          ││
│  │  💰 FUNDING: ETH 0.42% → SHORT opportunity (18:15:00)    ││
│  │  📈 VOLUME: BTC Accumulation at $42,150 (18:20:00)       ││
│  └───────────────────────────────────────────────────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. Strategy Discovery Panel

```
┌─────────────────────── STRATEGY DISCOVERY ──────────────────┐
│                                                              │
│  Current Cycle: 5 │ Next in: 20:00 │ Status: 🔍 SEARCHING  │
│                                                              │
│  Pipeline:                                                   │
│  ┌──────────┬──────────┬──────────┬──────────┐            │
│  │ Discover │ Backtest │ Validate │ Deploy   │            │
│  │   12     │    3     │    1     │   42     │            │
│  └──────────┴──────────┴──────────┴──────────┘            │
│                                                              │
│  Recent Discoveries:                                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 18:23:50 │ Machine Learning Commodity Trading         │ │
│  │          │ Source: Tavily │ Status: Queued             │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │ 18:23:35 │ Bollinger-Keltner Squeeze Breakout        │ │
│  │          │ Source: Tavily │ Status: Backtesting        │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │ 18:23:26 │ ML Volatility Trading Strategy            │ │
│  │          │ Source: Tavily │ Status: Validated ✅        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Stats:                                                      │
│  • Total Discovered: 156                                     │
│  • Approval Rate: 27.5%                                      │
│  • Best Source: TradingView (35% approval)                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

### 3. RBI Backtest Monitor

```
┌───────────────────── RBI BACKTEST ENGINE ────────────────────┐
│                                                               │
│  Active Backtest: Neural Network Volatility Forecasting      │
│  Progress: [████████████████░░░░] 80% (Iteration 8/10)      │
│                                                               │
│  Current Phase: 🔄 Optimization Loop                         │
│  Target Return: 50% │ Current Best: 24.3%                    │
│                                                               │
│  Debug Iterations:                                            │
│  ✅ Iteration 1: Syntax valid                                │
│  ✅ Iteration 2: Fixed import error                          │
│  ✅ Iteration 3: Fixed path error                            │
│  ✅ Iteration 4: Backtesting lib installed                   │
│  ✅ Iteration 5: Executing successfully                      │
│                                                               │
│  Multi-Config Testing:                                        │
│  ┌───────┬──────────┬────────┬──────┬─────────┐            │
│  │ Asset │Timeframe │ Win%   │ PF   │ Status  │            │
│  ├───────┼──────────┼────────┼──────┼─────────┤            │
│  │ BTC   │ 15m      │ 68.0%  │ 1.9  │ ✅ PASS │            │
│  │ ETH   │ 15m      │ 62.5%  │ 1.7  │ ✅ PASS │            │
│  │ SOL   │ 15m      │ 55.2%  │ 1.5  │ ✅ PASS │            │
│  └───────┴──────────┴────────┴──────┴─────────┘            │
│                                                               │
│  LLM Consensus Voting:                                        │
│  • DeepSeek R1:  ✅ APPROVE (High confidence)                │
│  • GPT-4:        ✅ APPROVE (Good metrics)                    │
│  • Claude 3.5:   ❌ REJECT (Drawdown concern)                │
│  Result: 2/3 APPROVED ✅                                      │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

### 4. Risk Management Dashboard

```
┌───────────────────── RISK MANAGEMENT ────────────────────────┐
│                                                               │
│  Portfolio Health: 🟢 HEALTHY                                │
│                                                               │
│  Exposure Gauge:                                              │
│  [████████████████████████░░░░░░] 77.8%                     │
│  $35,000 / $45,000 limit                                      │
│                                                               │
│  Risk Metrics:                                                │
│  ┌──────────┬─────────┬──────────┬──────────┐              │
│  │ Sharpe   │ Sortino │ Calmar   │ Max DD   │              │
│  │  1.65    │  2.12   │  1.89    │  3.2%    │              │
│  └──────────┴─────────┴──────────┴──────────┘              │
│                                                               │
│  Position Correlation:                                        │
│  ┌───────┬─────┬─────┬─────┐                                │
│  │       │ BTC │ ETH │ SOL │                                │
│  ├───────┼─────┼─────┼─────┤                                │
│  │ BTC   │ 1.0 │ 0.85│ 0.72│                                │
│  │ ETH   │ 0.85│ 1.0 │ 0.78│                                │
│  │ SOL   │ 0.72│ 0.78│ 1.0 │                                │
│  └───────┴─────┴─────┴─────┘                                │
│                                                               │
│  ⚠️  Alerts:                                                 │
│  • HIGH_CORRELATION: BTC-ETH (0.85) - Diversify              │
│  • POSITION_SIZE: Champion_833 at 28% (near 30% limit)       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

### 5. Market Signals Feed

```
┌──────────────────── MARKET SIGNALS FEED ─────────────────────┐
│                                                               │
│  🔴 LIVE │ Auto-refresh: ON │ Filter: All Signals            │
│                                                               │
│  18:28:33 🐋 WHALE ALERT                                     │
│  │ Symbol: BTC/USDT                                          │
│  │ OI Change: -22.53% → $2.96B                               │
│  │ Action: SELL signal                                       │
│  │ Confidence: 68%                                           │
│  └─ [Volume Agent] ✅ Confirmed by volume analysis           │
│                                                               │
│  18:27:33 🐋 WHALE ALERT                                     │
│  │ Symbol: BTC/USDT                                          │
│  │ OI Change: +55.92% → $3.72B                               │
│  │ Action: BUY signal                                        │
│  │ Confidence: 75%                                           │
│  └─ [Volume Agent] ✅ Confirmed - institutional accumulation │
│                                                               │
│  18:17:33 📊 SENTIMENT EXTREME                               │
│  │ Sentiment: BEARISH                                        │
│  │ Score: -0.86 (Extreme)                                    │
│  │ Confidence: 82%                                           │
│  └─ [Research Agent] ⚠️  Contrarian opportunity?            │
│                                                               │
│  18:15:00 💰 FUNDING OPPORTUNITY                             │
│  │ Symbol: ETH/USDT                                          │
│  │ Rate: 0.42% (8h)                                          │
│  │ Action: SHORT (funding arbitrage)                         │
│  │ Confidence: 68%                                           │
│  └─ [Risk Agent] ✅ Within risk limits                       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Real-Time Data Streaming

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'champion_update':
      updateChampionCard(data.champion);
      break;
    case 'new_signal':
      addSignalToFeed(data.signal);
      break;
    case 'backtest_progress':
      updateBacktestProgress(data.progress);
      break;
    case 'discovery_cycle':
      updateDiscoveryCycle(data.cycle);
      break;
  }
};
```

### WebSocket Message Types

#### 1. Champion Update
```json
{
  "type": "champion_update",
  "timestamp": "2025-11-21T18:30:00Z",
  "champion": {
    "id": "champion_1732211833",
    "bankroll": 11275.80,
    "profit_pct": 12.76,
    "last_trade": {
      "action": "BUY",
      "symbol": "BTC/USDT",
      "price": 42150.50,
      "size": 0.05,
      "pnl": +125.30
    }
  }
}
```

#### 2. New Signal
```json
{
  "type": "new_signal",
  "timestamp": "2025-11-21T18:30:15Z",
  "signal": {
    "source": "whale_agent",
    "signal_type": "LARGE_OI_CHANGE",
    "symbol": "btcusdt",
    "data": {
      "change_pct": 45.2,
      "oi_value": "$4.2B",
      "action": "BUY",
      "confidence": 0.78
    }
  }
}
```

#### 3. Backtest Progress
```json
{
  "type": "backtest_progress",
  "timestamp": "2025-11-21T18:30:20Z",
  "progress": {
    "strategy_name": "ML Volatility Trading",
    "phase": "optimization",
    "iteration": 6,
    "max_iterations": 10,
    "current_return": 32.5,
    "target_return": 50.0
  }
}
```

---

## Component Specifications

### Champion Card Component

```jsx
<ChampionCard champion={champion}>
  <Header>
    <StatusBadge status={champion.status} />
    <ChampionID>{champion.id}</ChampionID>
    <StrategyName>{champion.strategy_name}</StrategyName>
  </Header>
  
  <Metrics>
    <MetricRow>
      <Bankroll value={champion.bankroll} change={champion.profit_pct} />
      <WinRate value={champion.win_rate_trades} />
    </MetricRow>
    <MetricRow>
      <Trades total={champion.total_trades} today={champion.trades_today} />
      <Sharpe value={champion.sharpe_ratio} />
    </MetricRow>
  </Metrics>
  
  <MiniChart data={champion.equity_curve} />
  
  <Actions>
    <Button>View Details</Button>
    <Button>Performance</Button>
    {champion.status === 'ELITE' && <Button variant="primary">Trade Now</Button>}
  </Actions>
</ChampionCard>
```

**Styling Guidelines:**
- **ELITE**: Gold border, glow effect
- **QUALIFIED**: Silver border  
- **CHAMPION**: Bronze/grey border
- Green text for profits, red for losses
- Animated pulse on active trades

---

### Signal Card Component

```jsx
<SignalCard signal={signal}>
  <Icon type={signal.signal_type} animated />
  <Timestamp>{signal.detected_at}</Timestamp>
  
  <SignalHeader>
    <Type>{signal.signal_type}</Type>
    <Symbol>{signal.symbol}</Symbol>
  </SignalHeader>
  
  <SignalData>
    {signal.signal_type === 'WHALE' && (
      <>
        <OIChange value={signal.change_pct} />
        <OIValue>{signal.oi_value}</OIValue>
      </>
    )}
  </SignalData>
  
  <Action recommendation={signal.action} confidence={signal.confidence} />
  
  {signal.volume_confirmed && (
    <Badge variant="success">✅ Volume Confirmed</Badge>
  )}
</SignalCard>
```

**Color Coding:**
- 🐋 Whale: Blue
- 📊 Sentiment: Purple (green=bullish, red=bearish)
- 💰 Funding: Orange
- 📈 Volume: Green (accumulation) / Red (distribution)

---

### Backtest Progress Component

```jsx
<BacktestProgress backtest={backtest}>
  <Header>
    <StrategyName>{backtest.strategy_name}</StrategyName>
    <Phase>{backtest.phase}</Phase>
  </Header>
  
  <ProgressBar 
    current={backtest.iteration} 
    max={backtest.max_iterations}
    percentage={(backtest.iteration / backtest.max_iterations) * 100}
  />
  
  <Metrics>
    <Current return={backtest.current_return} />
    <Target return={backtest.target_return} />
    <Gap value={backtest.target_return - backtest.current_return} />
  </Metrics>
  
  <IterationLog>
    {backtest.iterations.map(iter => (
      <IterationItem 
        key={iter.number}
        status={iter.status}
        message={iter.message}
      />
    ))}
  </IterationLog>
</BacktestProgress>
```

---

## Data Models & Structures

### Champion Model
```typescript
interface Champion {
  id: string;
  status: 'CHAMPION' | 'QUALIFIED' | 'ELITE';
  bankroll: number;
  profit_pct: number;
  total_trades: number;
  trades_today: number;
  winning_days: number;
  total_days: number;
  win_rate_days: number;
  win_rate_trades: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  current_positions: number;
  strategy_name: string;
  real_trading_eligible: boolean;
  created_at: string;
  last_trade_at: string;
  equity_curve?: EquityPoint[];
  recent_trades?: Trade[];
}

interface EquityPoint {
  timestamp: string;
  equity: number;
}

interface Trade {
  timestamp: string;
  action: 'BUY' | 'SELL';
  symbol: string;
  price: number;
  size: number;
  pnl: number;
  win: boolean;
}
```

### Signal Model
```typescript
interface Signal {
  source: 'whale_agent' | 'sentiment_agent' | 'funding_agent' | 'volume_agent';
  signal_type: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  detected_at: string;
  data: SignalData;
  volume_confirmed?: boolean;
  research_validated?: boolean;
  risk_approved?: boolean;
}

interface WhaleSignalData {
  change_pct: number;
  oi_value: string;
  direction: 'UP' | 'DOWN';
}

interface SentimentSignalData {
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  score: number;
  magnitude: 'EXTREME' | 'HIGH' | 'MODERATE' | 'LOW';
  sources: string[];
}
```

---

## UI/UX Guidelines

### Color Palette

```css
:root {
  /* Status Colors */
  --status-running: #22c55e;
  --status-stopped: #ef4444;
  --status-warning: #f59e0b;
  
  /* Champion Tiers */
  --elite-gold: #fbbf24;
  --qualified-silver: #9ca3af;
  --champion-bronze: #d97706;
  
  /* Profit/Loss */
  --profit-green: #10b981;
  --loss-red: #ef4444;
  
  /* Signals */
  --whale-blue: #3b82f6;
  --sentiment-purple: #a855f7;
  --funding-orange: #f97316;
  --volume-teal: #14b8a6;
  
  /* Background */
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-card: #334155;
  
  /* Text */
  --text-primary: #f1f5f9;
  --text-secondary: #cbd5e1;
  --text-muted: #64748b;
}
```

### Typography

```css
/* Headers */
.dashboard-title {
  font-size: 2rem;
  font-weight: 700;
  color: var(--text-primary);
}

.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
}

/* Metrics */
.metric-value {
  font-size: 2.5rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.metric-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
```

### Animation Guidelines

```css
/* Status Pulse */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-running {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

/* Signal Entry */
@keyframes slideIn {
  from {
    transform: translateX(-100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

.signal-card {
  animation: slideIn 0.3s ease-out;
}

/* Elite Glow */
@keyframes glow {
  0%, 100% {
    box-shadow: 0 0 20px rgba(251, 191, 36, 0.5);
  }
  50% {
    box-shadow: 0 0 40px rgba(251, 191, 36, 0.8);
  }
}

.champion-elite {
  animation: glow 2s ease-in-out infinite;
}
```

---

## Update Frequencies

### Real-Time (WebSocket)
- Champion trades: Instant
- New signals: Instant
- Backtest progress: Every 5-10 seconds

### Polling (REST API)
- System status: Every 5 seconds
- Champions list: Every 10 seconds
- Discovery status: Every 30 seconds
- Risk analytics: Every 60 seconds

---

## Example Implementation

### React Dashboard (Simplified)

```jsx
import React, { useState, useEffect } from 'react';
import useWebSocket from 'react-use-websocket';

function APEXDashboard() {
  const [champions, setChampions] = useState([]);
  const [signals, setSignals] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  
  // WebSocket connection
  const { lastMessage } = useWebSocket('ws://localhost:8000/ws');
  
  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage !== null) {
      const data = JSON.parse(lastMessage.data);
      
      switch(data.type) {
        case 'champion_update':
          updateChampion(data.champion);
          break;
        case 'new_signal':
          addSignal(data.signal);
          break;
      }
    }
  }, [lastMessage]);
  
  // Fetch initial data
  useEffect(() => {
    fetchChampions();
    fetchSystemStatus();
    
    const interval = setInterval(() => {
      fetchSystemStatus();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);
  
  async function fetchChampions() {
    const res = await fetch('http://localhost:8000/api/champions');
    const data = await res.json();
    setChampions(data.champions);
  }
  
  async function fetchSystemStatus() {
    const res = await fetch('http://localhost:8000/api/system_status');
    const data = await res.json();
    setSystemStatus(data);
  }
  
  function updateChampion(updatedChampion) {
    setChampions(prev => prev.map(c => 
      c.id === updatedChampion.id ? updatedChampion : c
    ));
  }
  
  function addSignal(signal) {
    setSignals(prev => [signal, ...prev].slice(0, 50));
  }
  
  return (
    <div className="dashboard">
      <Header status={systemStatus} />
      
      <div className="grid">
        <div className="col-span-8">
          <ChampionsPanel champions={champions} />
        </div>
        
        <div className="col-span-4">
          <SignalsFeed signals={signals} />
        </div>
      </div>
      
      <div className="grid">
        <div className="col-span-6">
          <DiscoveryPanel />
        </div>
        
        <div className="col-span-6">
          <BacktestPanel />
        </div>
      </div>
    </div>
  );
}
```

---

## Summary

This frontend specification provides:

1. ✅ **Complete API documentation** for all endpoints
2. ✅ **Detailed dashboard layouts** with ASCII mockups
3. ✅ **Component specifications** with props and styling
4. ✅ **Data models** in TypeScript
5. ✅ **Real-time WebSocket** integration guide
6. ✅ **Color palette & animations** for consistent UI
7. ✅ **Example React implementation** to get started

The frontend should display:
- **All 8 agents' activities** in real-time
- **Complete champion lifecycle** from creation to ELITE
- **Strategy discovery pipeline** with live updates
- **RBI backtest progress** with detailed iterations
- **Market signals** from all 4 data sources (whale/sentiment/funding/volume)
- **Risk analytics** with portfolio health metrics
- **System health** with thread status and queue depths

This creates a **fully transparent, real-time view** of the entire APEX autonomous trading system.
