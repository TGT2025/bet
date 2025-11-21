# 🚀 APEX - Autonomous Profit EXtraction System

**Complete Monolithic Trading System with 5 Background Threads**

Version: 1.0  
Architecture: Single Monolith (1310 lines)  
Based on: E17FINAL + Moon-Dev Architecture  
Target: Fully Autonomous 24/7 Trading System

---

## 📋 System Overview

APEX is a fully autonomous trading system that runs 24/7 with ZERO human input after initial setup.

### One Command Launch
```bash
python apex.py
```

### Five Autonomous Threads

1. **Strategy Discovery** (30min cycles)
   - LLM-generated web search queries
   - Multi-source strategy extraction (TradingView, Medium, arXiv, GitHub)
   - Quality filtering
   - Automatic saving to strategy library

2. **RBI Backtest Engine** (validation pipeline)
   - DeepSeek R1 code generation
   - Multi-configuration testing (BTC/ETH/SOL × 3 timeframes × 3 periods)
   - LLM swarm consensus (DeepSeek + GPT-4 + Claude)
   - 2/3 votes required for approval

3. **Champion Manager** ($10K bankroll, 3-tier qualification)
   - CHAMPION → QUALIFIED → ELITE progression
   - Paper trading with realistic simulation
   - Auto-promotion based on performance
   - Real trading eligible after ELITE qualification

4. **Market Data Agents** (whale/sentiment/funding)
   - Whale Agent: Large transfer monitoring (>$1M)
   - Sentiment Agent: Social media sentiment analysis
   - Funding Agent: Perpetual futures funding rates

5. **API Server** (port 8000, real-time monitoring)
   - FastAPI dashboard
   - Champion status tracking
   - System health monitoring
   - Queue status visualization

### Communication
- In-memory thread-safe queues
- No database dependencies
- Real-time signal propagation

### LLM Swarm
- Multi-model consensus voting
- DeepSeek R1 (primary)
- GPT-4 (validation)
- Claude 3.5 Sonnet (validation)
- Gemini (optional)

---

## 🔑 Required API Keys

### Core LLMs (Swarm Consensus)
```bash
export DEEPSEEK_API_KEY="sk-..."      # Primary RBI agent
export OPENAI_API_KEY="sk-..."        # GPT-4 validation
export ANTHROPIC_API_KEY="sk-ant-..."  # Claude validation
export GOOGLE_API_KEY="..."           # Gemini validation (optional)
```

### Web Search (Choose One)
```bash
export TAVILY_API_KEY="tvly-..."      # Primary search
# OR
export PERPLEXITY_API_KEY="pplx-..."  # Alternative search
```

### Exchange
```bash
export HTX_API_KEY="..."
export HTX_SECRET="..."
```

### Optional Data Sources
```bash
export TWITTER_API_KEY="..."          # Sentiment analysis
export ETHERSCAN_API_KEY="..."        # On-chain data
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Launch APEX
```bash
python apex.py
```

### 4. Access Monitoring Dashboard
```
http://localhost:8000
```

---

## 📊 API Endpoints

### System Status
```
GET /
GET /api/system_status
```

### Champions
```
GET /api/champions
```

Response:
```json
{
  "champions": [
    {
      "id": "champion_1763570000",
      "status": "QUALIFIED",
      "bankroll": 11250.50,
      "profit_pct": 12.51,
      "total_trades": 75,
      "trades_today": 8,
      "winning_days": 5,
      "total_days": 7,
      "win_rate_days": 71.4,
      "real_trading_eligible": false
    }
  ],
  "summary": {
    "total_champions": 12,
    "elite": 1,
    "qualified": 3,
    "champions": 8,
    "total_bankroll": 125000.0,
    "total_profit": 8500.50
  }
}
```

---

## 🎯 Champion Qualification System

### CHAMPION (Entry Tier)
- Starting bankroll: $10,000
- Default leverage: 5x
- Qualification requirements:
  - ✅ 3 days minimum
  - ✅ 50 trades minimum
  - ✅ 60% winning days
  - ✅ 8% profit

### QUALIFIED (Middle Tier)
- Upgraded from CHAMPION
- Qualification requirements:
  - ✅ 14 days minimum
  - ✅ 200 trades minimum
  - ✅ 65% winning days
  - ✅ 25% profit

### ELITE (Top Tier)
- Upgraded from QUALIFIED
- **Real Trading Eligible** ✅
- Proven consistent performance

---

## 📈 Performance Targets

- **Strategy Discovery**: 50+ strategies/day
- **Backtest Approval Rate**: 20-30% (quality filter)
- **Champion Creation**: 1-3 new champions/day
- **Trade Frequency**: 5+ trades/day per champion
- **Real Trading Eligible**: 1 ELITE champion within 14-21 days

---

## 🏗️ Architecture

### Monolithic Design
- Single file (`apex.py`)
- No external modules
- Self-contained logic
- Dynamically creates all folders

### Directory Structure (Auto-Created)
```
/
├── apex.py              # Main monolith (1310 lines)
├── .env                 # API keys
├── requirements.txt     # Dependencies
├── logs/                # Execution logs
├── strategy_library/    # Discovered strategies
├── checkpoints/         # System state
└── data/               # Market data
```

### Thread Safety
- `threading.Lock()` for champion data
- `queue.Queue()` for inter-thread communication
- No shared state mutations
- Daemon threads for automatic cleanup

---

## 🔧 Configuration

Edit variables in `Config` class within `apex.py`:

### Strategy Discovery
```python
DISCOVERY_INTERVAL_MINUTES = 30
DISCOVERY_QUERIES_PER_CYCLE = 15
```

### RBI Backtest
```python
MIN_WIN_RATE = 0.55  # 55%
MIN_PROFIT_FACTOR = 1.5
MAX_DRAWDOWN = 0.20  # 20%
MIN_SHARPE_RATIO = 1.0
MIN_TRADES = 50
CONSENSUS_REQUIRED_VOTES = 2  # Out of 3
```

### Champion Manager
```python
STARTING_BANKROLL = 10000.0
DEFAULT_LEVERAGE = 5.0
TRADE_INTERVAL_MINUTES = 5
RISK_PER_TRADE_PERCENT = 0.02  # 2%
```

### Market Data Agents
```python
WHALE_MIN_AMOUNT_USD = 1_000_000  # $1M
SENTIMENT_EXTREME_THRESHOLD = 0.7
FUNDING_RATE_THRESHOLD = 0.001  # 0.1%
```

---

## 🔒 Security

- API keys stored in environment variables only
- Never logs sensitive data
- Paper trading only until ELITE status
- Rate limiting on all external APIs
- Exception handling prevents crashes
- Thread monitoring with auto-restart

---

## 🐛 Troubleshooting

### Missing API Keys
```
❌ Missing required API keys: DEEPSEEK_API_KEY, OPENAI_API_KEY
```
**Solution**: Set all required keys in `.env` file

### No Search Results
```
⚠️ No search API key (TAVILY or PERPLEXITY) - strategy discovery will be limited
```
**Solution**: Add TAVILY_API_KEY or PERPLEXITY_API_KEY to `.env`

### Thread Crashed
Check logs in `logs/apex_*.log` for error details

### Port 8000 Already in Use
```
❌ API server error: Address already in use
```
**Solution**: Change `API_PORT` in Config or kill process on port 8000

---

## 📦 Dependencies

- Python 3.8+
- numpy
- pandas
- openai
- anthropic
- google-generativeai
- python-dotenv
- fastapi
- uvicorn
- requests

---

## 🎓 Based On

- **E17FINAL**: Monolithic architecture pattern
- **Moon-Dev AI Agents**: RBI v3, WebSearch, Whale, Sentiment, Funding agents
- **Backtesting.py**: Strategy testing framework

---

## 📝 License

Proprietary - All Rights Reserved

---

## 👨‍💻 Author

Built for autonomous 24/7 profit extraction

---

## 🚨 Disclaimer

**This software is for educational purposes only.**

Trading cryptocurrencies and financial derivatives carries substantial risk of loss. Past performance does not guarantee future results. Only trade with capital you can afford to lose.

The authors and contributors are not responsible for any financial losses incurred through use of this software.

---

## 📞 Support

For issues and questions, check the logs in `logs/apex_*.log`

---

**🚀 Ready to Launch APEX!**

```bash
python apex.py
```

Access dashboard: http://localhost:8000

---
