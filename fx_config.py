"""
OANDA FX Configuration
Complete configuration for OANDA forex trading system
"""

import os
from typing import Dict, List

# =========================================================================================
# OANDA API CREDENTIALS
# =========================================================================================

# Load from environment variables
OANDA_API_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_PRACTICE = os.getenv("OANDA_PRACTICE", "True").lower() == "true"

# API URLs
OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"
OANDA_LIVE_URL = "https://api-fxtrade.oanda.com"
OANDA_STREAM_PRACTICE = "https://stream-fxpractice.oanda.com"
OANDA_STREAM_LIVE = "https://stream-fxtrade.oanda.com"

# =========================================================================================
# TRADING INSTRUMENTS
# =========================================================================================

FX_INSTRUMENTS = {
    "majors": [
        "EUR_USD",  # Euro / US Dollar
        "GBP_USD",  # British Pound / US Dollar
        "USD_JPY",  # US Dollar / Japanese Yen
        "USD_CHF",  # US Dollar / Swiss Franc
        "AUD_USD",  # Australian Dollar / US Dollar
        "USD_CAD",  # US Dollar / Canadian Dollar
        "NZD_USD",  # New Zealand Dollar / US Dollar
    ],
    "minors": [
        "EUR_GBP",  # Euro / British Pound
        "EUR_JPY",  # Euro / Japanese Yen
        "EUR_CHF",  # Euro / Swiss Franc
        "EUR_AUD",  # Euro / Australian Dollar
        "EUR_CAD",  # Euro / Canadian Dollar
        "GBP_JPY",  # British Pound / Japanese Yen
        "GBP_CHF",  # British Pound / Swiss Franc
        "GBP_AUD",  # British Pound / Australian Dollar
        "CHF_JPY",  # Swiss Franc / Japanese Yen
        "AUD_JPY",  # Australian Dollar / Japanese Yen
        "AUD_CAD",  # Australian Dollar / Canadian Dollar
        "NZD_JPY",  # New Zealand Dollar / Japanese Yen
    ],
    "exotics": [
        "USD_TRY",  # US Dollar / Turkish Lira
        "USD_ZAR",  # US Dollar / South African Rand
        "USD_MXN",  # US Dollar / Mexican Peso
        "USD_SGD",  # US Dollar / Singapore Dollar
        "USD_HKD",  # US Dollar / Hong Kong Dollar
        "EUR_TRY",  # Euro / Turkish Lira
        "GBP_ZAR",  # British Pound / South African Rand
    ]
}

# Default instruments to trade (start with majors only)
DEFAULT_INSTRUMENTS = FX_INSTRUMENTS["majors"][:3]  # EUR_USD, GBP_USD, USD_JPY

# =========================================================================================
# TIMEFRAMES / GRANULARITY
# =========================================================================================

# OANDA granularity codes
OANDA_GRANULARITIES = {
    "S5": "5 second",
    "S10": "10 second",
    "S15": "15 second",
    "S30": "30 second",
    "M1": "1 minute",
    "M2": "2 minute",
    "M4": "4 minute",
    "M5": "5 minute",
    "M10": "10 minute",
    "M15": "15 minute",
    "M30": "30 minute",
    "H1": "1 hour",
    "H2": "2 hour",
    "H3": "3 hour",
    "H4": "4 hour",
    "H6": "6 hour",
    "H8": "8 hour",
    "H12": "12 hour",
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly",
}

# Trading timeframes by strategy type
TIMEFRAMES = {
    "scalping": "M5",     # 5-minute scalping
    "intraday": "M15",    # 15-minute intraday
    "swing": "H1",        # 1-hour swing trading
    "position": "H4",     # 4-hour position trading
    "daily": "D",         # Daily charts
}

# Default timeframe
DEFAULT_TIMEFRAME = "M15"

# =========================================================================================
# RISK MANAGEMENT PARAMETERS
# =========================================================================================

RISK_CONFIG = {
    # Position sizing
    "max_leverage": 20,                    # Maximum leverage (conservative)
    "position_size_pct": 0.01,            # Risk 1% of account per trade
    "max_position_pct": 0.02,             # Maximum 2% in single position
    "max_total_exposure_pct": 0.10,       # Maximum 10% total exposure
    
    # Stop loss / Take profit (in pips)
    "default_sl_pips": 20,                # Default 20 pip stop loss
    "default_tp_pips": 40,                # Default 40 pip take profit
    "min_sl_pips": 10,                    # Minimum 10 pip SL
    "max_sl_pips": 100,                   # Maximum 100 pip SL
    "min_tp_pips": 15,                    # Minimum 15 pip TP
    "max_tp_pips": 200,                   # Maximum 200 pip TP
    
    # Risk/Reward
    "min_rr_ratio": 1.5,                  # Minimum 1.5:1 reward:risk
    "target_rr_ratio": 2.0,               # Target 2:1 reward:risk
    
    # Spread limits (in pips)
    "max_spread_pips": {
        "majors": 2.0,                    # Max 2 pip spread for majors
        "minors": 4.0,                    # Max 4 pip spread for minors
        "exotics": 10.0,                  # Max 10 pip spread for exotics
    },
    
    # Daily limits
    "max_daily_trades": 20,               # Max 20 trades per day
    "max_daily_loss_pct": 0.05,           # Max 5% daily loss
    "max_consecutive_losses": 3,          # Stop after 3 consecutive losses
    
    # Position limits
    "max_open_positions": 5,              # Max 5 concurrent positions
    "max_correlated_positions": 2,        # Max 2 highly correlated positions
    
    # Margin requirements
    "min_margin_pct": 0.30,               # Maintain 30% minimum margin
    "margin_call_pct": 0.20,              # Margin call at 20%
}

# =========================================================================================
# FOREX SESSION TIMES (UTC)
# =========================================================================================

FOREX_SESSIONS = {
    "SYDNEY": {
        "start_hour": 21,
        "end_hour": 6,
        "description": "Sydney Session",
        "active_pairs": ["AUD_USD", "NZD_USD", "AUD_JPY"]
    },
    "TOKYO": {
        "start_hour": 0,
        "end_hour": 9,
        "description": "Tokyo/Asian Session",
        "active_pairs": ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]
    },
    "LONDON": {
        "start_hour": 8,
        "end_hour": 17,
        "description": "London/European Session",
        "active_pairs": ["EUR_USD", "GBP_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY"]
    },
    "NEW_YORK": {
        "start_hour": 13,
        "end_hour": 22,
        "description": "New York/US Session",
        "active_pairs": ["EUR_USD", "GBP_USD", "USD_CAD", "USD_JPY"]
    }
}

# Session overlaps (highest liquidity)
SESSION_OVERLAPS = {
    "LONDON_NY": {
        "start_hour": 13,
        "end_hour": 17,
        "description": "London/New York Overlap - Highest Volatility",
        "recommended": True
    },
    "SYDNEY_TOKYO": {
        "start_hour": 0,
        "end_hour": 6,
        "description": "Sydney/Tokyo Overlap",
        "recommended": False
    }
}

# =========================================================================================
# CURRENCY PAIR CHARACTERISTICS
# =========================================================================================

PAIR_CHARACTERISTICS = {
    "EUR_USD": {
        "pip_value_per_10k": 1.0,         # $1 per pip for 10,000 units
        "typical_spread": 0.8,            # Typical spread in pips
        "avg_daily_range": 80,            # Average daily range in pips
        "volatility": "medium",
        "best_sessions": ["LONDON", "NEW_YORK"],
        "correlation_group": "USD"
    },
    "GBP_USD": {
        "pip_value_per_10k": 1.0,
        "typical_spread": 1.0,
        "avg_daily_range": 120,
        "volatility": "high",
        "best_sessions": ["LONDON", "NEW_YORK"],
        "correlation_group": "USD"
    },
    "USD_JPY": {
        "pip_value_per_10k": 0.91,        # Approx (varies with price)
        "typical_spread": 0.9,
        "avg_daily_range": 70,
        "volatility": "medium",
        "best_sessions": ["TOKYO", "NEW_YORK"],
        "correlation_group": "JPY"
    },
    "USD_CHF": {
        "pip_value_per_10k": 1.1,
        "typical_spread": 1.2,
        "avg_daily_range": 70,
        "volatility": "low",
        "best_sessions": ["LONDON", "NEW_YORK"],
        "correlation_group": "CHF"
    },
    "AUD_USD": {
        "pip_value_per_10k": 1.0,
        "typical_spread": 1.0,
        "avg_daily_range": 80,
        "volatility": "medium",
        "best_sessions": ["SYDNEY", "TOKYO"],
        "correlation_group": "Commodity"
    },
    "USD_CAD": {
        "pip_value_per_10k": 0.79,
        "typical_spread": 1.5,
        "avg_daily_range": 75,
        "volatility": "medium",
        "best_sessions": ["NEW_YORK"],
        "correlation_group": "Commodity"
    },
    "NZD_USD": {
        "pip_value_per_10k": 1.0,
        "typical_spread": 1.5,
        "avg_daily_range": 70,
        "volatility": "medium",
        "best_sessions": ["SYDNEY", "TOKYO"],
        "correlation_group": "Commodity"
    }
}

# =========================================================================================
# ECONOMIC CALENDAR - HIGH IMPACT EVENTS
# =========================================================================================

HIGH_IMPACT_EVENTS = {
    "NFP": {
        "description": "Non-Farm Payrolls (US Jobs Report)",
        "frequency": "Monthly - 1st Friday",
        "impact": "EXTREME",
        "affected_pairs": ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", "AUD_USD"],
        "avoid_minutes_before": 30,
        "avoid_minutes_after": 60
    },
    "FOMC": {
        "description": "Federal Reserve Meeting",
        "frequency": "8 times per year",
        "impact": "EXTREME",
        "affected_pairs": ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF"],
        "avoid_minutes_before": 30,
        "avoid_minutes_after": 120
    },
    "ECB": {
        "description": "European Central Bank Meeting",
        "frequency": "8 times per year",
        "impact": "EXTREME",
        "affected_pairs": ["EUR_USD", "EUR_GBP", "EUR_JPY", "EUR_CHF"],
        "avoid_minutes_before": 30,
        "avoid_minutes_after": 120
    },
    "BOE": {
        "description": "Bank of England Meeting",
        "frequency": "8 times per year",
        "impact": "EXTREME",
        "affected_pairs": ["GBP_USD", "EUR_GBP", "GBP_JPY"],
        "avoid_minutes_before": 30,
        "avoid_minutes_after": 120
    },
    "CPI": {
        "description": "Consumer Price Index (Inflation)",
        "frequency": "Monthly",
        "impact": "HIGH",
        "affected_pairs": "ALL",
        "avoid_minutes_before": 15,
        "avoid_minutes_after": 30
    },
    "GDP": {
        "description": "Gross Domestic Product",
        "frequency": "Quarterly",
        "impact": "HIGH",
        "affected_pairs": "ALL",
        "avoid_minutes_before": 15,
        "avoid_minutes_after": 30
    },
    "EMPLOYMENT": {
        "description": "Employment/Unemployment Data",
        "frequency": "Monthly",
        "impact": "HIGH",
        "affected_pairs": "Currency-specific",
        "avoid_minutes_before": 15,
        "avoid_minutes_after": 30
    }
}

# =========================================================================================
# CURRENCY CORRELATIONS (Typical values -1 to 1)
# =========================================================================================

TYPICAL_CORRELATIONS = {
    ("EUR_USD", "GBP_USD"): 0.85,        # Highly positive correlation
    ("EUR_USD", "USD_CHF"): -0.90,       # Highly negative (inverse)
    ("EUR_USD", "USD_JPY"): -0.65,       # Moderately negative
    ("GBP_USD", "EUR_GBP"): -0.70,       # Negative correlation
    ("AUD_USD", "NZD_USD"): 0.92,        # Very high positive (both commodity currencies)
    ("AUD_USD", "USD_CAD"): -0.75,       # Negative
    ("USD_JPY", "EUR_JPY"): 0.75,        # Positive (JPY is common)
    ("USD_CAD", "EUR_CAD"): 0.80,        # Positive (CAD is common)
}

# =========================================================================================
# SWAP/ROLLOVER INFORMATION
# =========================================================================================

# Note: Actual swap rates change daily and should be fetched from OANDA API
# These are just for reference

SWAP_INFO = {
    "rollover_time": "22:00 GMT (5pm ET)",  # When rollover occurs
    "triple_swap_day": "Wednesday",          # Triple swap on Wednesday (weekend)
    "description": "Swap rates are the interest differential between currency pairs",
    "note": "Fetch actual rates from OANDA API get_instruments() endpoint"
}

# =========================================================================================
# STRATEGY CONFIGURATION
# =========================================================================================

STRATEGY_CONFIG = {
    # Strategy types
    "strategy_types": ["scalping", "intraday", "swing"],
    
    # Alpha sources for FX
    "fx_alpha_sources": [
        "interest_rate_differential",
        "central_bank_divergence",
        "pivot_point_reversals",
        "session_breakouts",
        "fibonacci_clusters",
        "round_number_bounces",
        "bid_ask_imbalance",
        "safe_haven_flows",
        "commodity_currency_correlation",
    ],
    
    # Technical indicators suitable for FX
    "technical_indicators": [
        "moving_averages",       # 20 EMA, 50 SMA, 200 SMA
        "rsi",                   # Relative Strength Index
        "macd",                  # MACD
        "bollinger_bands",       # Bollinger Bands
        "fibonacci",             # Fibonacci retracements
        "pivot_points",          # Daily/weekly pivots
        "atr",                   # Average True Range
        "stochastic",            # Stochastic Oscillator
    ],
    
    # Min trades enforcement
    "min_trades_global": 15,
    "min_symbols_trading": 2,
    "min_per_tier_trades": {
        "low": 3,
        "medium": 5,
        "high": 7
    }
}

# =========================================================================================
# RESEARCH CONFIGURATION
# =========================================================================================

RESEARCH_CONFIG = {
    "instruments": DEFAULT_INSTRUMENTS,
    "periods": [DEFAULT_TIMEFRAME],
    "target_period": DEFAULT_TIMEFRAME,
    "max_iterations": 50,
    "alpha_threshold": 0.35,
    "min_sharpe_ratio": 1.2,
    "max_drawdown_pct": 0.15,
}

# =========================================================================================
# HELPER FUNCTIONS
# =========================================================================================

def get_pip_size(instrument: str) -> float:
    """
    Get pip size for an instrument
    
    Most pairs: 0.0001 (4th decimal)
    JPY pairs: 0.01 (2nd decimal)
    """
    if "JPY" in instrument or "HUF" in instrument:
        return 0.01
    return 0.0001

def get_instrument_category(instrument: str) -> str:
    """Get category of instrument (majors/minors/exotics)"""
    for category, instruments in FX_INSTRUMENTS.items():
        if instrument in instruments:
            return category
    return "unknown"

def get_max_spread(instrument: str) -> float:
    """Get maximum acceptable spread for instrument"""
    category = get_instrument_category(instrument)
    return RISK_CONFIG["max_spread_pips"].get(category, 10.0)

def is_trading_hours(current_hour_utc: int, session: str = None) -> bool:
    """Check if current time is within trading session"""
    if session:
        session_info = FOREX_SESSIONS.get(session)
        if not session_info:
            return False
        start = session_info["start_hour"]
        end = session_info["end_hour"]
    else:
        # Check if within any major session
        for sess_info in FOREX_SESSIONS.values():
            start = sess_info["start_hour"]
            end = sess_info["end_hour"]
            if start < end:
                if start <= current_hour_utc < end:
                    return True
            else:  # Crosses midnight
                if current_hour_utc >= start or current_hour_utc < end:
                    return True
        return False
    
    # Handle session crossing midnight
    if start < end:
        return start <= current_hour_utc < end
    else:
        return current_hour_utc >= start or current_hour_utc < end

def get_current_session(current_hour_utc: int) -> str:
    """Get current forex session"""
    for session_name, session_info in FOREX_SESSIONS.items():
        if is_trading_hours(current_hour_utc, session_name):
            return session_name
    return "OFF_HOURS"

# =========================================================================================
# EXPORT CONFIG GETTER
# =========================================================================================

def get_config() -> Dict:
    """Get complete configuration"""
    return {
        "oanda": {
            "api_key": OANDA_API_KEY,
            "account_id": OANDA_ACCOUNT_ID,
            "practice": OANDA_PRACTICE,
            "base_url": OANDA_PRACTICE_URL if OANDA_PRACTICE else OANDA_LIVE_URL,
        },
        "instruments": DEFAULT_INSTRUMENTS,
        "timeframe": DEFAULT_TIMEFRAME,
        "risk": RISK_CONFIG,
        "strategy": STRATEGY_CONFIG,
        "research": RESEARCH_CONFIG,
    }

def get_risk_config() -> Dict:
    """Get risk configuration"""
    return RISK_CONFIG.copy()

def get_research_config() -> Dict:
    """Get research configuration"""
    return RESEARCH_CONFIG.copy()
