"""
🚀 APEX Configuration
Central configuration for the APEX trading system
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =========================================================================================
# SYSTEM CONFIGURATION
# =========================================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
STRATEGY_LIBRARY_DIR = PROJECT_ROOT / "strategy_library"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Ensure directories exist
for directory in [DATA_DIR, LOGS_DIR, STRATEGY_LIBRARY_DIR, CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =========================================================================================
# API KEYS
# =========================================================================================

# Core LLMs
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Web Search
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

# Exchange
HTX_API_KEY = os.getenv("HTX_API_KEY", "")
HTX_SECRET = os.getenv("HTX_SECRET", "")

# Optional
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")

# =========================================================================================
# LLM MODELS
# =========================================================================================

# Strategy Discovery
DISCOVERY_MODEL = "gpt-4"
DISCOVERY_TEMPERATURE = 0.7
DISCOVERY_MAX_TOKENS = 4000

# RBI Backtest Engine
RBI_PRIMARY_MODEL = "deepseek-reasoner"  # DeepSeek R1
RBI_VALIDATION_MODELS = ["gpt-4", "claude-3-5-sonnet-20240620"]
RBI_TEMPERATURE = 0.3
RBI_MAX_TOKENS = 8000

# Consensus voting
CONSENSUS_REQUIRED_VOTES = 2  # Out of 3 LLMs

# =========================================================================================
# THREAD 1: STRATEGY DISCOVERY
# =========================================================================================

DISCOVERY_INTERVAL_MINUTES = 30  # Run every 30 minutes
DISCOVERY_QUERIES_PER_CYCLE = 15  # Number of search queries per cycle

# Search sources
SEARCH_SOURCES = [
    "TradingView strategies",
    "Medium crypto quant",
    "arXiv quantitative finance",
    "GitHub trading strategies",
    "Quantpedia database"
]

# =========================================================================================
# THREAD 2: RBI BACKTEST ENGINE
# =========================================================================================

# Auto-debug settings
MAX_DEBUG_ITERATIONS = 10
BACKTEST_TIMEOUT_SECONDS = 300

# Multi-configuration testing
TEST_ASSETS = ["BTC", "ETH", "SOL"]
TEST_TIMEFRAMES = ["15m", "1H", "4H"]
TEST_PERIODS_DAYS = [30, 60, 90]
TEST_FEE_PERCENT = 0.1  # 0.1% realistic slippage

# Approval criteria
MIN_WIN_RATE = 0.55  # 55%
MIN_PROFIT_FACTOR = 1.5
MAX_DRAWDOWN = 0.20  # 20%
MIN_SHARPE_RATIO = 1.0
MIN_TRADES = 50

# =========================================================================================
# THREAD 3: CHAMPION MANAGER
# =========================================================================================

# Champion lifecycle
STARTING_BANKROLL = 10000.0  # $10K USD
DEFAULT_LEVERAGE = 5.0  # 5x leverage
TRADE_INTERVAL_MINUTES = 5  # Check for signals every 5 minutes

# Risk management
RISK_PER_TRADE_PERCENT = 0.02  # 2% max risk per trade
MAX_POSITION_PERCENT = 0.30  # 30% max position size

# Qualification thresholds
CHAMPION_TO_QUALIFIED = {
    "min_days": 3,
    "min_trades": 50,
    "min_win_rate_days": 0.60,  # 60% winning days
    "min_profit_percent": 8.0    # 8% profit
}

QUALIFIED_TO_ELITE = {
    "min_days": 14,
    "min_trades": 200,
    "min_win_rate_days": 0.65,  # 65% winning days
    "min_profit_percent": 25.0   # 25% profit
}

# =========================================================================================
# THREAD 4: MARKET DATA AGENTS
# =========================================================================================

# Whale agent
WHALE_CHECK_INTERVAL_SECONDS = 60  # Check every minute
WHALE_MIN_AMOUNT_USD = 1_000_000  # $1M minimum
WHALE_CONFIDENCE = 0.7

# Sentiment agent
SENTIMENT_CHECK_INTERVAL_SECONDS = 300  # Every 5 minutes
SENTIMENT_EXTREME_THRESHOLD = 0.7  # Absolute value
SENTIMENT_CONFIDENCE_MAX = 0.9

# Funding agent
FUNDING_CHECK_INTERVAL_SECONDS = 3600  # Every hour
FUNDING_RATE_THRESHOLD = 0.001  # 0.1% threshold
FUNDING_CONFIDENCE_MULTIPLIER = 100

# =========================================================================================
# THREAD 5: API SERVER
# =========================================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = False  # Set to True for development

# =========================================================================================
# HTX EXCHANGE CONFIG
# =========================================================================================

HTX_BASE_URL = "https://api.huobi.pro"
HTX_WEBSOCKET_URL = "wss://api.huobi.pro/ws"

# =========================================================================================
# SYSTEM HEALTH
# =========================================================================================

THREAD_CHECK_INTERVAL_SECONDS = 60  # Check thread health every minute
HEARTBEAT_TIMEOUT_SECONDS = 300  # 5 minutes without heartbeat = dead thread

# =========================================================================================
# LOGGING
# =========================================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
