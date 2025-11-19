# E17 FINAL → OANDA FX TRADING SYSTEM
## Complete Architecture Analysis & Migration Plan

---

## EXECUTIVE SUMMARY

**Objective**: Adapt the current crypto trading system (e17final) to work with OANDA FX (forex) trading while preserving the core agent-based alpha hunting architecture.

**Current State**: HTX crypto exchange with USDT trading pairs
**Target State**: OANDA FX broker with currency pairs (majors/minors/exotics)

**Approach**: Strategic refactoring keeping proven components, replacing exchange-specific parts

---

## PART 1: WHAT TO KEEP (CORE FRAMEWORK)

### ✅ 1.1 Agent Architecture & AI System
**Keep 100% - These are exchange-agnostic**

- **Super Reasoner** - Plan creation and strategy architecture
- **Expert Coder** - Code generation for strategies
- **Enforcer** - Min-trades validation and quality checks
- **System Auditor** - Performance validation
- **Agent Ecosystem** - ResearchQuant, ExecutionQuant, RiskQuant agents

**Why**: These agents operate on abstract trading concepts, not exchange specifics.

### ✅ 1.2 Monitoring & Infrastructure
**Keep 100% - Platform independent**

- **Checkpointing System** (CheckpointManager)
- **Heartbeat Monitor** (HeartbeatMonitor)
- **Watchdog System** (WatchdogMonitor)
- **Agent Memory Persistence** (AgentMemoryManager)
- **Flask Monitoring API** (all /api/* endpoints)
- **Monolith Alpha Engine** (agent_activity_tracker, API endpoints)
- **Logging System** (setup_enhanced_logging)

**Why**: These are infrastructure components independent of trading venue.

### ✅ 1.3 Core Trading Logic
**Keep with modifications**

- **Strategy generation framework** - The loop that creates strategies
- **Backtesting framework** - Testing strategies on data
- **Alpha scoring system** - Evaluating strategy quality
- **Champion promotion logic** - Selecting best strategies
- **Multi-iteration loop** - Continuous improvement

**Modification needed**: Adapt to FX-specific metrics (pips, spreads, swaps)

---

## PART 2: WHAT TO CHANGE (EXCHANGE-SPECIFIC)

### 🔄 2.1 Exchange Client (CRITICAL)
**Replace entirely**: HTXClient → OANDAClient

**Current (HTX)**:
```python
class HTXClient:
    def fetch_klines(symbol, interval, limit)  # OHLCV bars
    def get_market_data()                      # Current prices
    def place_order()                          # Market/limit orders
```

**New (OANDA)**:
```python
class OANDAClient:
    def __init__(api_key, account_id, practice=True)
    def get_candles(instrument, granularity, count)  # M1,M5,H1,D
    def get_pricing(instruments)                     # Bid/ask spreads
    def create_order(instrument, units, type)        # Market/limit/stop
    def get_account_summary()                        # Balance, margin
    def get_open_positions()                         # Current positions
    def get_open_trades()                            # Active trades
```

**OANDA API Specifics**:
- REST API v20: https://api-fxtrade.oanda.com (live) / https://api-fxpractice.oanda.com (demo)
- Authentication: Bearer token (your API key)
- Rate limits: Varies by endpoint (typically 120 req/min)

### 🔄 2.2 Trading Pairs / Instruments
**Transform completely**

**Current (Crypto)**:
- BTCUSDT, ETHUSDT, BNBUSDT, etc.
- Base currency + USDT
- No concept of "direction" (always vs USDT)

**New (FX)**:
- EUR_USD, GBP_USD, USD_JPY, AUD_USD (majors)
- EUR_GBP, EUR_JPY, GBP_JPY (crosses)
- USD_TRY, USD_ZAR, EUR_PLN (exotics)
- **Bidirectional**: Can trade EUR_USD or USD_EUR
- **Pip calculation**: Different for each pair
- **Correlation aware**: EUR_USD vs GBP_USD often correlated

**Instrument Categories**:
```python
FX_MAJORS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD"
]

FX_MINORS = [
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD",
    "GBP_JPY", "GBP_CHF", "AUD_JPY", "NZD_JPY"
]

FX_EXOTICS = [
    "USD_TRY", "USD_ZAR", "USD_MXN", "EUR_TRY",
    "GBP_ZAR", "AUD_MXN"
]
```

### 🔄 2.3 Data Feed Structure
**Adapt to OANDA format**

**Current (HTX)**:
```python
{
    "btcusdt": pd.DataFrame({
        "timestamp": [...],
        "open": [...],
        "high": [...],
        "low": [...],
        "close": [...],
        "volume": [...]
    })
}
```

**New (OANDA)**:
```python
{
    "EUR_USD": pd.DataFrame({
        "time": [...],          # UTC timestamp
        "bid_o": [...],         # Bid OHLC
        "bid_h": [...],
        "bid_l": [...],
        "bid_c": [...],
        "ask_o": [...],         # Ask OHLC
        "ask_h": [...],
        "ask_l": [...],
        "ask_c": [...],
        "volume": [...],        # Tick volume
        "complete": [...]       # Bar completion flag
    })
}
```

**Key Differences**:
1. Bid/Ask separation (spreads matter in FX!)
2. Time is in RFC3339 format
3. Volume is "tick volume" not actual traded volume
4. Need to calculate mid price: (bid + ask) / 2

### 🔄 2.4 Order Execution
**Completely different**

**HTX (Crypto)**:
- Order types: market, limit
- Fees: Maker/taker (0.1-0.2%)
- Slippage: Usually minimal for liquid pairs

**OANDA (FX)**:
- Order types: market, limit, stop, stop-loss, take-profit, trailing-stop
- Spreads: Variable (0.5-3 pips for majors, more for exotics)
- **No commission** on standard accounts (profit from spreads)
- **Leverage**: Up to 50:1 (US), 30:1 (EU), 500:1 (other)
- **Swaps/Rollover**: Interest charges for holding overnight
- **Margin requirements**: Must maintain margin ratio

**Position Sizing**:
```python
# Crypto: Trade in base currency units
order = {"symbol": "BTCUSDT", "quantity": 0.01, "price": 43000}

# FX: Trade in "units" of base currency
order = {
    "instrument": "EUR_USD",
    "units": 10000,  # 10,000 EUR (= 0.1 lots)
    "type": "MARKET"
}
```

---

## PART 3: FX-SPECIFIC FEATURES TO ADD

### 🆕 3.1 Forex Sessions & Timing
**Critical for FX strategies**

```python
FOREX_SESSIONS = {
    "SYDNEY": {"open": "21:00 UTC", "close": "06:00 UTC"},
    "TOKYO": {"open": "00:00 UTC", "close": "09:00 UTC"},
    "LONDON": {"open": "08:00 UTC", "close": "17:00 UTC"},
    "NEW_YORK": {"open": "13:00 UTC", "close": "22:00 UTC"}
}

# Session overlaps = highest liquidity
LONDON_NY_OVERLAP = "13:00-17:00 UTC"  # Most volatile period
```

**Why it matters**:
- Volatility varies by session
- EUR/GBP most active during London
- USD/JPY most active during Tokyo/NY
- Avoid trading during Asian session for EUR pairs (low liquidity)

### 🆕 3.2 Economic Calendar Integration
**Essential for FX**

Major events to track:
- **NFP** (Non-Farm Payrolls) - 1st Friday each month
- **FOMC** (Federal Reserve meetings)
- **ECB** (European Central Bank announcements)
- **CPI** (Consumer Price Index)
- **GDP** releases
- **Central bank interest rate decisions**

**Strategy adjustment**:
```python
def avoid_high_impact_events(current_time):
    """Don't trade 30 min before/after major news"""
    upcoming_events = get_economic_calendar()
    for event in upcoming_events:
        if event['impact'] == 'HIGH':
            if abs(current_time - event['time']) < timedelta(minutes=30):
                return True
    return False
```

### 🆕 3.3 Currency Correlation Analysis
**FX pairs are correlated**

```python
TYPICAL_CORRELATIONS = {
    ("EUR_USD", "GBP_USD"): 0.85,   # Highly positive
    ("EUR_USD", "USD_CHF"): -0.90,  # Highly negative (inverse)
    ("AUD_USD", "NZD_USD"): 0.92,   # Very similar (commodity currencies)
    ("USD_JPY", "EUR_JPY"): 0.75,   # Positive (JPY is common)
}

def check_correlation_risk(positions):
    """Avoid over-concentration in correlated pairs"""
    if has_long("EUR_USD") and has_long("GBP_USD"):
        # Double exposure to USD weakness!
        reduce_position_size()
```

### 🆕 3.4 Pip Value & P&L Calculation
**Different for each pair**

```python
def calculate_pip_value(instrument, units, account_currency="USD"):
    """
    EUR_USD: 1 pip = 0.0001
    USD_JPY: 1 pip = 0.01
    EUR_GBP: 1 pip = 0.0001
    """
    if "JPY" in instrument or "HUF" in instrument:
        pip_size = 0.01
    else:
        pip_size = 0.0001
    
    # For EUR_USD with 10,000 units:
    # 1 pip movement = 10,000 * 0.0001 = $1
    return units * pip_size
```

### 🆕 3.5 Swap/Rollover Management
**Holding overnight costs money**

```python
def calculate_overnight_swap(instrument, units, is_long, days=1):
    """
    Tom/Next rates (interest differential between currencies)
    
    EUR_USD long: Pay if EUR rate < USD rate, Earn if EUR rate > USD rate
    """
    swap_rates = get_swap_rates(instrument)  # From OANDA API
    
    if is_long:
        swap = swap_rates['long'] * units * days
    else:
        swap = swap_rates['short'] * units * days
    
    return swap  # Can be positive (earn) or negative (pay)
```

---

## PART 4: PROMPT MODIFICATIONS FOR FX

### 🔄 4.1 Super Reasoner Prompts
**Current (Crypto)**:
```
"Analyze HTX USDT trading pairs for momentum, volatility patterns..."
```

**New (FX)**:
```
You are analyzing OANDA FX markets. Consider:
- Currency pair categories (majors/minors/exotics)
- Forex session timing (London/NY/Tokyo/Sydney)
- Interest rate differentials and carry trade opportunities
- Economic calendar events (NFP, FOMC, ECB, CPI)
- Currency correlations (EUR/USD vs GBP/USD)
- Pip-based profit targets (10-30 pips for scalping, 50-200 for swing)
- Spread costs (wider for exotics)
- Swap rates for overnight holds

Focus on FX-specific alpha sources:
- Interest rate differentials (carry trades)
- Central bank policy divergence
- Economic data surprises
- Risk-on/risk-off flows
- Technical levels (FX respects support/resistance)
```

### 🔄 4.2 Strategy Generation Prompts
**Current**:
```
"Generate momentum strategy for BTCUSDT using RSI and MACD..."
```

**New**:
```
Generate FX strategy for [EUR_USD, GBP_USD, USD_JPY]:
- Use pip-based targets (TP: 20 pips, SL: 10 pips)
- Account for spread (2 pips for EUR_USD)
- Avoid trading during major news (NFP, FOMC)
- Consider session volatility (London most active for EUR pairs)
- Check currency correlation (don't double-up on USD exposure)
- Include swap consideration for multi-day holds

Technical indicators suitable for FX:
- Pivot points (widely watched in FX)
- Fibonacci retracements (respected in FX)
- Moving averages (20 EMA, 50 SMA, 200 SMA)
- RSI (30/70 levels)
- Bollinger Bands (2 std dev)
- ATR for volatility (pip-based stops)
```

### 🔄 4.3 Alpha Sources for FX
**Replace crypto alphas with FX alphas**

**Crypto Alpha Sources** (remove):
- Funding rate arbitrage
- Whale wallet tracking
- Dark pool indicator
- Gamma scalping (options)

**FX Alpha Sources** (add):
```python
FX_ALPHA_SOURCES = [
    # Fundamental
    "interest_rate_differential",      # Carry trade
    "central_bank_divergence",         # Policy differences
    "economic_surprise_index",         # Data vs expectations
    "risk_sentiment_indicators",       # VIX, safe havens
    
    # Technical  
    "pivot_point_reversals",           # Daily/weekly pivots
    "session_breakouts",               # London/NY open
    "fibonacci_clusters",              # Multi-timeframe fibs
    "round_number_bounces",            # 1.1000, 1.2000, etc.
    
    # Microstructure
    "bid_ask_imbalance",              # Order flow
    "time_of_day_patterns",           # Session-specific
    "spread_widening_signals",        # Liquidity stress
    
    # Macro
    "commodity_currency_correlation",  # AUD/CAD vs oil/metals
    "safe_haven_flows",               # JPY/CHF in risk-off
    "cross_pair_arbitrage"            # EUR/USD vs EUR/GBP vs GBP/USD
]
```

---

## PART 5: CODE CHANGES REQUIRED

### 5.1 New OANDA Client Module
**File**: `oanda_client.py`

```python
import requests
from datetime import datetime, timedelta
import pandas as pd

class OANDAClient:
    def __init__(self, api_key: str, account_id: str, practice: bool = True):
        self.api_key = api_key
        self.account_id = account_id
        self.practice = practice
        
        if practice:
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_candles(self, instrument: str, granularity: str = "M5", 
                    count: int = 500) -> pd.DataFrame:
        """
        Fetch OHLC candles
        
        instrument: "EUR_USD", "GBP_USD", etc.
        granularity: "M1", "M5", "M15", "H1", "H4", "D"
        count: Number of candles (max 5000)
        """
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": count,
            "price": "MBA"  # Mid, Bid, Ask
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        candles = response.json()['candles']
        
        # Convert to DataFrame
        data = []
        for candle in candles:
            if candle['complete']:  # Only completed candles
                data.append({
                    'time': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume']),
                    'bid_close': float(candle['bid']['c']),
                    'ask_close': float(candle['ask']['c']),
                })
        
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'])
        return df
    
    def get_pricing(self, instruments: list) -> dict:
        """Get current bid/ask prices"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": ",".join(instruments)}
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        
        return response.json()['prices']
    
    def create_market_order(self, instrument: str, units: int, 
                           sl_pips: float = None, tp_pips: float = None):
        """
        Create market order
        
        units: Positive for long, negative for short
               10000 = 0.1 lots, 100000 = 1 lot
        sl_pips: Stop loss in pips
        tp_pips: Take profit in pips
        """
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        
        order_spec = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units)
        }
        
        # Add stop loss/take profit
        if sl_pips:
            order_spec["stopLossOnFill"] = {
                "distance": str(sl_pips * self._get_pip_size(instrument))
            }
        
        if tp_pips:
            order_spec["takeProfitOnFill"] = {
                "distance": str(tp_pips * self._get_pip_size(instrument))
            }
        
        data = {"order": order_spec}
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        
        return response.json()
    
    def _get_pip_size(self, instrument: str) -> float:
        """Get pip size for instrument"""
        if "JPY" in instrument or "HUF" in instrument:
            return 0.01
        return 0.0001
    
    def get_account_summary(self):
        """Get account balance, margin, etc."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()['account']
    
    def get_open_positions(self):
        """Get all open positions"""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/openPositions"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()['positions']
```

### 5.2 FX-Specific Configuration
**File**: `fx_config.py`

```python
# OANDA credentials (from environment variables)
OANDA_API_KEY = os.getenv("OANDA_API_KEY")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID")
OANDA_PRACTICE = os.getenv("OANDA_PRACTICE", "True") == "True"

# Trading instruments
FX_INSTRUMENTS = {
    "majors": [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
        "AUD_USD", "USD_CAD", "NZD_USD"
    ],
    "minors": [
        "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD",
        "GBP_JPY", "GBP_CHF", "AUD_JPY"
    ],
    "exotics": [
        "USD_TRY", "USD_ZAR", "USD_MXN"
    ]
}

# Default to trade majors only
DEFAULT_INSTRUMENTS = FX_INSTRUMENTS["majors"]

# Timeframes (OANDA granularity)
TIMEFRAMES = {
    "scalping": "M5",    # 5-minute
    "intraday": "M15",   # 15-minute
    "swing": "H1",       # 1-hour
    "position": "H4"     # 4-hour
}

# Risk parameters (FX-specific)
RISK_CONFIG = {
    "max_leverage": 20,              # Max 20:1 leverage
    "max_position_pct": 0.02,        # Max 2% account per trade
    "max_daily_loss_pct": 0.05,      # Max 5% daily loss
    "max_spread_pips": {
        "majors": 2,                 # Max 2 pip spread for majors
        "minors": 4,                 # Max 4 pip spread for minors
        "exotics": 10                # Max 10 pip spread for exotics
    },
    "default_sl_pips": 20,           # Default 20 pip stop loss
    "default_tp_pips": 40,           # Default 40 pip take profit (2:1 R:R)
    "min_rr_ratio": 1.5,             # Minimum 1.5:1 reward:risk
}

# Session times (UTC)
FOREX_SESSIONS = {
    "SYDNEY": {"start": 21, "end": 6},
    "TOKYO": {"start": 0, "end": 9},
    "LONDON": {"start": 8, "end": 17},
    "NEW_YORK": {"start": 13, "end": 22}
}

# Economic calendar (high impact events to avoid)
HIGH_IMPACT_EVENTS = [
    "NFP",              # Non-Farm Payrolls
    "FOMC",             # Fed meeting
    "ECB",              # ECB meeting
    "BOE",              # Bank of England
    "CPI",              # Inflation data
    "GDP",              # Growth data
    "EMPLOYMENT"        # Jobs data
]
```

### 5.3 Modify Main System
**File**: `e17final_fx.py` (new FX version)

Key changes:
1. Replace HTXClient with OANDAClient
2. Update market data fetching for bid/ask spreads
3. Modify strategy generation prompts for FX
4. Add FX-specific validation (spread checks, session filters)
5. Update position sizing for units instead of quantity
6. Add swap rate consideration
7. Implement economic calendar filtering

---

## PART 6: IMPLEMENTATION PHASES

### Phase 1: Foundation (Week 1)
- [ ] Create OANDAClient class
- [ ] Test API connectivity with practice account
- [ ] Implement candle fetching and data normalization
- [ ] Create FX configuration module
- [ ] Test basic order placement (practice account)

### Phase 2: Core Adaptation (Week 2)
- [ ] Fork e17final → e17final_fx
- [ ] Replace exchange client integration
- [ ] Update market data structures (bid/ask)
- [ ] Modify strategy generation prompts for FX
- [ ] Add FX-specific alpha sources

### Phase 3: FX Features (Week 3)
- [ ] Implement session detection and filtering
- [ ] Add economic calendar integration
- [ ] Build currency correlation checker
- [ ] Implement pip-based P&L calculations
- [ ] Add swap rate consideration

### Phase 4: Testing & Validation (Week 4)
- [ ] Backtest on historical FX data
- [ ] Paper trade with practice account
- [ ] Validate spread handling
- [ ] Test multi-session strategies
- [ ] Verify risk management (leverage, margin)

### Phase 5: Production (Week 5)
- [ ] Connect to live account (with safeguards)
- [ ] Monitor performance
- [ ] Fine-tune alpha thresholds for FX
- [ ] Document FX-specific learnings

---

## PART 7: RISKS & MITIGATIONS

### Risk 1: Spread Costs
**Problem**: FX spreads can eat into profits (2-10 pips)
**Mitigation**: 
- Filter out high-spread periods
- Require min profit target > 2x spread
- Trade during high liquidity sessions

### Risk 2: Leverage Danger
**Problem**: High leverage can wipe account quickly
**Mitigation**:
- Start with max 10:1 leverage
- Strict position sizing (max 2% per trade)
- Hard stop losses on every trade
- Daily loss limit (5% max)

### Risk 3: News Volatility
**Problem**: Major news can cause 50-100 pip spikes
**Mitigation**:
- Integrate economic calendar
- No trading 30 min before/after high-impact events
- Widen stops during news or stay flat

### Risk 4: Swap Charges
**Problem**: Holding overnight can be expensive
**Mitigation**:
- Check swap rates before holding
- Prefer positive swap pairs for multi-day
- Consider closing before 5pm ET (rollover time)

### Risk 5: Correlation Risk
**Problem**: Trading EUR/USD and GBP/USD = double USD exposure
**Mitigation**:
- Calculate correlation matrix
- Limit correlated positions (max 2 pairs >0.7 correlation)
- Diversify across different base currencies

---

## PART 8: EXPECTED OUTCOMES

### Performance Targets (FX vs Crypto)
**Crypto (Current)**:
- Volatility: High (5-15% daily swings)
- Trades: 20-50 per day
- Win rate target: 55-60%
- Avg trade: 0.5-2% profit

**FX (Target)**:
- Volatility: Lower (0.5-2% daily ranges)
- Trades: 10-30 per day (more selective)
- Win rate target: 60-65% (better predictability)
- Avg trade: 10-40 pips (0.1-0.4% with leverage)

### Advantages of FX
✅ 24/5 market (more predictable hours)
✅ Higher liquidity (especially majors)
✅ Lower slippage
✅ Leveraged returns without holding assets
✅ More fundamental data (economic indicators)
✅ Established technical patterns (more respected)

### Challenges of FX
⚠️ Lower volatility (need leverage for returns)
⚠️ Spread costs on every trade
⚠️ Overnight swap charges
⚠️ Complex risk management (margin calls)
⚠️ News-driven volatility spikes
⚠️ Correlation complexity

---

## PART 9: RECOMMENDED STARTING CONFIGURATION

```python
# Start conservative for FX
INITIAL_FX_CONFIG = {
    "account_type": "practice",           # Use demo first!
    "instruments": ["EUR_USD", "GBP_USD"], # Start with 2 majors
    "timeframe": "M15",                   # 15-minute scalping
    "max_leverage": 10,                   # Conservative leverage
    "position_size_pct": 0.01,            # 1% per trade
    "stop_loss_pips": 20,                 # 20 pip stops
    "take_profit_pips": 40,               # 40 pip targets
    "max_daily_trades": 10,               # Limit overtrading
    "max_open_positions": 2,              # Max 2 concurrent trades
    "trading_sessions": ["LONDON", "NEW_YORK"],  # Most liquid
    "avoid_news": True,                   # Skip news events
}
```

---

## CONCLUSION

This is a **significant architectural change** but the core agent system is perfectly suited for it. The key is:

1. **Keep**: Agent framework, monitoring, Monolith tracking
2. **Replace**: Exchange client, data structures, order execution
3. **Add**: FX-specific features (sessions, swaps, correlations, economic calendar)
4. **Adapt**: Prompts and alpha sources for FX market characteristics

**Estimated effort**: 3-4 weeks for full implementation
**Risk level**: Medium (practice account mitigates financial risk)
**Success probability**: High (agent architecture is exchange-agnostic)

**Next step**: Review this analysis and approve before I begin Phase 1 implementation.
