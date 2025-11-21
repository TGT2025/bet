# 🚀 APEX - Autonomous Profit EXtraction System

**APEX** = Best of E17FINAL + Moon-Dev AI Agents

## 🎯 What is APEX?

APEX is a fully autonomous trading system that:
1. **Discovers** profitable strategies from the web (TradingView, papers, GitHub)
2. **Validates** them using RBI backtest engine with auto-debugging
3. **Promotes** winners to champion tier with $10K virtual bankroll
4. **Qualifies** champions through 3-tier progression (CHAMPION → QUALIFIED → ELITE)
5. **Monitors** real-time market data (whale movements, sentiment, funding rates)

## 🏗️ Architecture

**Single Process, 5 Background Threads:**

```
python apex/apex_main.py
```

### Thread 1: Strategy Discovery Agent
- **Purpose**: Autonomous strategy research
- **Frequency**: Every 30 minutes
- **Sources**: TradingView, Medium, arXiv, Quantpedia, GitHub
- **Output**: `strategy_discovery_queue`

### Thread 2: RBI Backtest Engine
- **Purpose**: Validate strategies with Moon-Dev's proven engine
- **Features**: Auto-debug (10 iterations), multi-config testing
- **Quality Gates**: PF >1.5, WR >55%, DD <20%
- **Output**: `validated_strategy_queue`

### Thread 3: Champion Manager (E17FINAL Core)
- **Purpose**: Paper & live trading with qualification system
- **Bankroll**: $10K starting capital per champion
- **Tiers**: 
  - 🥉 CHAMPION: Entry (1+ trade)
  - 🥈 QUALIFIED: 3 days, 50+ trades, 60%+ win days, 8%+ profit
  - 🥇 ELITE: 14 days, 200+ trades, 65%+ win days, 25%+ profit
- **Trading**: HTX futures with 3-10x leverage

### Thread 4: Market Data Agents
- **whale_agent**: Monitors large HTX transfers (>1M USDT)
- **sentiment_agent**: Twitter/Reddit sentiment analysis
- **funding_agent**: Perpetual funding rate tracking
- **Output**: `market_data_queue`

### Thread 5: API Server
- **Port**: 8000
- **Endpoints**:
  - `GET /api/champions` - List all champions
  - `GET /api/champions/{id}` - Champion details
  - `GET /api/strategy_discovery` - Recent discoveries
  - `GET /api/backtest_results` - RBI validation results
  - `GET /api/market_signals` - Whale/sentiment/funding data
  - `GET /api/system_status` - Health metrics

## 🔑 Required API Keys

```bash
# Core LLMs (Multi-model consensus)
export DEEPSEEK_API_KEY="sk-..."          # Primary RBI agent
export OPENAI_API_KEY="sk-..."            # GPT-4 validation
export ANTHROPIC_API_KEY="sk-ant-..."     # Claude validation
export GOOGLE_API_KEY="..."               # Gemini (optional)

# Web Search
export TAVILY_API_KEY="tvly-..."          # Primary
# OR
export PERPLEXITY_API_KEY="pplx-..."      # Alternative

# Exchange
export HTX_API_KEY="..."
export HTX_SECRET="..."

# Optional Data Sources
export TWITTER_API_KEY="..."              # Sentiment
export ETHERSCAN_API_KEY="..."            # On-chain data
```

## 🚀 Quick Start

1. **Set API keys** (see above)
2. **Install dependencies**:
   ```bash
   pip install -r apex/requirements.txt
   ```
3. **Run APEX**:
   ```bash
   python apex/apex_main.py
   ```
4. **Monitor dashboard**:
   ```
   http://localhost:8000
   ```

## 📊 What Makes APEX Unique?

### From E17FINAL:
✅ Champion progression system (CHAMPION → QUALIFIED → ELITE)  
✅ $10K bankroll tracking per champion  
✅ 3-tier qualification with accelerated timeline (3d/14d)  
✅ HTX futures leverage trading (3-10x)  
✅ Real trading path (ELITE = eligible for real money)  
✅ Continuous 24/7 operation  

### From Moon-Dev:
✅ Fully automated strategy discovery (websearch agent)  
✅ RBI backtest engine with auto-debugging (10 iterations)  
✅ Multi-config testing (20+ combinations)  
✅ Proven backtesting.py library integration  
✅ Multi-model LLM swarm (DeepSeek + GPT-4 + Claude)  

### APEX Exclusive:
✅ Real-time market data agents (whale/sentiment/funding)  
✅ In-memory queue communication between all threads  
✅ Single monolithic file (easy deployment)  
✅ Zero human input after initial setup  
✅ Production-ready error handling  

## 📈 Champion Lifecycle

```
Strategy Discovery → RBI Backtest → CHAMPION ($10K) → Paper Trade
                                         ↓
                                    3 Days + Metrics
                                         ↓
                                    QUALIFIED
                                         ↓
                                    14 Days + Metrics
                                         ↓
                                    ELITE (Real Trading Eligible ✅)
```

## 🔍 Monitoring

### Real-time Metrics:
- Active champions count
- Total trades executed today
- Win rate distribution
- P&L by champion
- Strategy discovery rate
- Backtest success rate
- Market signal frequency

### API Examples:

```bash
# List all champions
curl http://localhost:8000/api/champions

# Get specific champion
curl http://localhost:8000/api/champions/champion_1234567890

# View recent strategy discoveries
curl http://localhost:8000/api/strategy_discovery

# Check system health
curl http://localhost:8000/api/system_status
```

## 🛠️ Architecture Details

### Queue Communication:
- `strategy_discovery_queue`: Thread 1 → Thread 2
- `validated_strategy_queue`: Thread 2 → Thread 3
- `market_data_queue`: Thread 4 → Thread 3

### Thread Safety:
- All queues use Python's `queue.Queue` (thread-safe)
- Champion data protected by threading locks
- API responses use read-only views

### Error Handling:
- Each thread has independent try/except blocks
- Threads restart on failure (max 3 attempts)
- Comprehensive logging for debugging
- Graceful shutdown on SIGTERM/SIGINT

## 📦 File Structure

```
apex/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── apex_main.py              # Main monolithic file (all 5 threads)
├── .env.example              # API key template
└── data/                     # Runtime data
    ├── champions/            # Champion state
    ├── strategies/           # Discovered strategies
    ├── backtests/           # RBI results
    └── market_data/         # Whale/sentiment/funding
```

## 🎯 Performance Targets

- **Strategy Discovery**: 10+ new strategies/day
- **Backtest Success**: >30% pass quality gates
- **Champion Creation**: 2-5 new champions/day
- **Paper Trading**: 5+ trades/day per champion
- **Qualification Rate**: 40% reach QUALIFIED in 3 days
- **Elite Rate**: 20% reach ELITE in 14 days

## ⚠️ Important Notes

1. **This is paper trading** until champions reach ELITE tier
2. **Real money trading** requires manual approval even for ELITE
3. **All trades** are simulated until you enable real trading mode
4. **API keys** are required for full functionality
5. **Backup your data** regularly (champions/, strategies/, backtests/)

## 🆘 Troubleshooting

### Common Issues:

**"No strategies discovered"**
- Check TAVILY_API_KEY or PERPLEXITY_API_KEY is set
- Verify internet connection
- Check logs for rate limiting

**"Backtest failures"**
- Check DEEPSEEK_API_KEY is valid
- Ensure sufficient API credits
- Review auto-debug logs

**"No champions promoted"**
- Strategies may not meet quality gates (PF <1.5, WR <55%)
- Increase backtest iterations
- Review strategy discovery sources

**"API server won't start"**
- Port 8000 may be in use: `lsof -i :8000`
- Check firewall settings
- Try different port in config

## 📝 License

This is a hybrid system combining:
- E17FINAL (proprietary champion system)
- Moon-Dev AI Agents (open source: https://github.com/moondevonyt/moon-dev-ai-agents)

Please respect both licenses and attribution.

## 🙏 Credits

- **E17FINAL**: Champion progression, bankroll management, HTX integration
- **Moon-Dev**: RBI backtest engine, strategy discovery, multi-model LLM
- **APEX**: Integration architecture, market data agents, unified system

---

**Built with ❤️ for autonomous trading**
