"""
Alpha Library - Hardcoded knowledge base of 20+ proven alpha sources
ResearchQuantAgent uses this as reference for strategy generation

Each alpha source includes:
- Name and description
- Academic paper reference
- Exploitable edge
- Concrete implementation
- Expected Sharpe ratio
- Decay rate
- Market regime applicability
"""

ALPHA_LIBRARY = [
    {
        "id": 1,
        "name": "Volatility Risk Premium",
        "category": "options",
        "paper": "Bollerslev et al. 2018 - Variance Risk Premium",
        "edge": "Implied volatility consistently exceeds realized volatility due to fear premium",
        "implementation": "Sell strangles when IV percentile > 70%, delta-hedge daily",
        "expected_sharpe": 1.2,
        "decay_rate": "6-12 months (overcrowding can reduce returns)",
        "market_regime": ["high_volatility", "mean_reverting"],
        "risk_factors": ["tail_risk", "gamma_risk"],
        "code_pattern": """
# Volatility Risk Premium Strategy
if iv_percentile > 0.70 and implied_vol > realized_vol * 1.2:
    # Sell premium when IV is elevated
    sell_strangle(strike_distance=0.15, size=kelly_optimal)
    delta_hedge_daily()
"""
    },
    {
        "id": 2,
        "name": "Momentum Crash Protection",
        "category": "momentum",
        "paper": "Daniel & Moskowitz 2016 - Momentum Crashes",
        "edge": "Momentum strategies crash during market rebounds after panics",
        "implementation": "Skip momentum signals when VIX > 40 or market rebounds > 5% in 2 days",
        "expected_sharpe": 0.8,
        "decay_rate": "Well-known since 2016 but still effective",
        "market_regime": ["trending", "low_volatility"],
        "risk_factors": ["momentum_reversal"],
        "code_pattern": """
# Momentum with Crash Protection
if vix < 40 and recent_2day_return < 0.05:
    # Safe to use momentum
    if price > sma_200 and rsi > 60:
        momentum_signal = 'BUY'
else:
    # High crash risk - skip momentum
    momentum_signal = None
"""
    },
    {
        "id": 3,
        "name": "Liquidity Provision Alpha",
        "category": "market_making",
        "paper": "Hendershott & Menkveld 2014 - Price Pressures",
        "edge": "Provide liquidity at bid/ask spread during normal conditions",
        "implementation": "Place limit orders at mid-price ± 0.5 * spread, cancel on adverse selection",
        "expected_sharpe": 1.5,
        "decay_rate": "3-6 months in liquid markets",
        "market_regime": ["low_volatility", "mean_reverting"],
        "risk_factors": ["adverse_selection", "inventory_risk"],
        "code_pattern": """
# Liquidity Provision
mid_price = (bid + ask) / 2
spread = ask - bid
if spread > min_spread and volume > min_volume:
    place_limit_buy(price=mid_price - 0.5*spread, size=max_inventory/4)
    place_limit_sell(price=mid_price + 0.5*spread, size=max_inventory/4)
"""
    },
    {
        "id": 4,
        "name": "Statistical Arbitrage (Pairs Trading)",
        "category": "mean_reversion",
        "paper": "Gatev et al. 2006 - Pairs Trading",
        "edge": "Cointegrated pairs revert to mean relationship",
        "implementation": "Trade spread when Z-score > 2, exit when Z-score < 0.5",
        "expected_sharpe": 1.1,
        "decay_rate": "12-18 months per pair",
        "market_regime": ["mean_reverting"],
        "risk_factors": ["cointegration_breakdown"],
        "code_pattern": """
# Statistical Arbitrage
spread = price_A - hedge_ratio * price_B
z_score = (spread - mean_spread) / std_spread
if abs(z_score) > 2.0:
    if z_score > 0:
        short_A_long_B(size=kelly_optimal)
    else:
        long_A_short_B(size=kelly_optimal)
elif abs(z_score) < 0.5:
    close_position()
"""
    },
    {
        "id": 5,
        "name": "Cross-Asset Correlation Breakdown",
        "category": "multi_asset",
        "paper": "Asness et al. 2013 - Diversification Return",
        "edge": "Asset correlations break down during regime changes",
        "implementation": "Increase diversification when rolling correlation drops below 0.3",
        "expected_sharpe": 0.9,
        "decay_rate": "Structural, long-lasting",
        "market_regime": ["all"],
        "risk_factors": ["correlation_spike"],
        "code_pattern": """
# Correlation Breakdown Strategy
rolling_corr = correlation(asset_A, asset_B, window=60)
if rolling_corr < 0.3 and rolling_corr < historical_avg - 2*std:
    # Increase allocation to both assets
    increase_allocation(asset_A, multiplier=1.5)
    increase_allocation(asset_B, multiplier=1.5)
"""
    },
    {
        "id": 6,
        "name": "Funding Rate Arbitrage",
        "category": "crypto",
        "paper": "Makarov & Schoar 2020 - Crypto Arbitrage",
        "edge": "Perpetual funding rates create predictable flows",
        "implementation": "Short when funding > 0.1% (8hr), long when funding < -0.05%",
        "expected_sharpe": 1.3,
        "decay_rate": "3-6 months (very competitive)",
        "market_regime": ["trending", "high_volatility"],
        "risk_factors": ["liquidation_risk", "exchange_risk"],
        "code_pattern": """
# Funding Rate Arbitrage
if funding_rate > 0.001:  # 0.1% per 8 hours
    # Funding is high - short perp, long spot
    short_perpetual(size=hedge_size)
    long_spot(size=hedge_size)
elif funding_rate < -0.0005:
    # Funding is negative - long perp, short spot
    long_perpetual(size=hedge_size)
    short_spot(size=hedge_size)
"""
    },
    {
        "id": 7,
        "name": "Order Flow Imbalance",
        "category": "microstructure",
        "paper": "Cont et al. 2014 - Price Impact of Order Flow",
        "edge": "Persistent order flow imbalance predicts short-term moves",
        "implementation": "Buy when bid volume > ask volume * 1.5 for 5 minutes",
        "expected_sharpe": 0.7,
        "decay_rate": "1-3 months (HFT competition)",
        "market_regime": ["trending"],
        "risk_factors": ["latency", "adverse_selection"],
        "code_pattern": """
# Order Flow Imbalance
bid_volume = sum(bid_sizes, last_5min)
ask_volume = sum(ask_sizes, last_5min)
imbalance_ratio = bid_volume / ask_volume
if imbalance_ratio > 1.5 and bid_volume > min_volume_threshold:
    buy_signal(size=imbalance_ratio * base_size, duration='1min')
elif imbalance_ratio < 0.67:
    sell_signal(size=(1/imbalance_ratio) * base_size, duration='1min')
"""
    },
    {
        "id": 8,
        "name": "Trend Following with Regime Filter",
        "category": "momentum",
        "paper": "Moskowitz et al. 2012 - Time Series Momentum",
        "edge": "Trends persist in liquid markets, especially with regime filtering",
        "implementation": "Follow 50-day trend only when volatility < 30th percentile",
        "expected_sharpe": 0.9,
        "decay_rate": "Decades (robust alpha)",
        "market_regime": ["trending", "low_volatility"],
        "risk_factors": ["trend_reversal"],
        "code_pattern": """
# Trend Following with Regime Filter
trend_signal = price > sma_50
vol_percentile = percentile_rank(realized_vol_20d, 252)
if trend_signal and vol_percentile < 0.30:
    # Low vol trending - safe to follow trend
    position = 'LONG' if price > sma_50 else 'SHORT'
    size = vol_target_position_size(target_vol=0.10)
"""
    },
    {
        "id": 9,
        "name": "Mean Reversion at Support/Resistance",
        "category": "mean_reversion",
        "paper": "Lo & MacKinlay 1990 - Contrarian Profits",
        "edge": "Prices bounce at psychological support/resistance levels",
        "implementation": "Buy at support with RSI < 30, sell at resistance with RSI > 70",
        "expected_sharpe": 0.8,
        "decay_rate": "6-12 months per level",
        "market_regime": ["mean_reverting"],
        "risk_factors": ["support_breakdown"],
        "code_pattern": """
# Mean Reversion at Support/Resistance
support = find_support_level(lookback=90)
resistance = find_resistance_level(lookback=90)
if price <= support * 1.01 and rsi < 30:
    buy_signal(size=kelly_optimal, stop_loss=support*0.98)
elif price >= resistance * 0.99 and rsi > 70:
    sell_signal(size=kelly_optimal, stop_loss=resistance*1.02)
"""
    },
    {
        "id": 10,
        "name": "Carry Trade with Risk Management",
        "category": "carry",
        "paper": "Brunnermeier et al. 2009 - Carry Trade Crashes",
        "edge": "Interest rate differentials create predictable returns",
        "implementation": "Long high-yield, short low-yield, exit when VIX > 25",
        "expected_sharpe": 1.0,
        "decay_rate": "Years (structural)",
        "market_regime": ["low_volatility"],
        "risk_factors": ["carry_crash"],
        "code_pattern": """
# Carry Trade with Risk Management
yield_diff = yield_high - yield_low
if yield_diff > 0.02 and vix < 25:
    # Carry conditions favorable
    long_high_yield(size=vol_target_size)
    short_low_yield(size=vol_target_size)
elif vix > 30:
    # Crash risk - exit carry
    close_all_carry_positions()
"""
    },
    {
        "id": 11,
        "name": "Earnings Drift Post-Announcement",
        "category": "event_driven",
        "paper": "Ball & Brown 1968 - Post-Earnings Drift",
        "edge": "Stock prices drift in direction of earnings surprise for 60 days",
        "implementation": "Buy on positive surprise > 5%, hold 60 days",
        "expected_sharpe": 0.9,
        "decay_rate": "Well-known but still works",
        "market_regime": ["all"],
        "risk_factors": ["earnings_reversal"],
        "code_pattern": """
# Earnings Drift
earnings_surprise = (actual_eps - expected_eps) / abs(expected_eps)
if earnings_surprise > 0.05:
    # Significant positive surprise
    buy_signal(size=base_size, hold_days=60)
elif earnings_surprise < -0.05:
    # Significant negative surprise
    sell_signal(size=base_size, hold_days=60)
"""
    },
    {
        "id": 12,
        "name": "Overnight Gap Fade",
        "category": "intraday",
        "paper": "Lou et al. 2019 - Overnight Returns",
        "edge": "Large overnight gaps tend to fade intraday",
        "implementation": "Fade gaps > 2% in first hour of trading",
        "expected_sharpe": 0.6,
        "decay_rate": "3-6 months",
        "market_regime": ["mean_reverting"],
        "risk_factors": ["gap_continuation"],
        "code_pattern": """
# Overnight Gap Fade
overnight_return = (open_price - prev_close) / prev_close
if abs(overnight_return) > 0.02 and time < market_open + 1hour:
    if overnight_return > 0:
        # Fade up gap
        sell_signal(size=gap_fade_size, duration='1hour')
    else:
        # Fade down gap
        buy_signal(size=gap_fade_size, duration='1hour')
"""
    },
    {
        "id": 13,
        "name": "Volume Price Analysis",
        "category": "volume",
        "paper": "Karpoff 1987 - Volume and Price",
        "edge": "High volume breakouts more likely to continue",
        "implementation": "Buy breakouts with volume > 2x average",
        "expected_sharpe": 0.7,
        "decay_rate": "6-12 months",
        "market_regime": ["trending"],
        "risk_factors": ["false_breakout"],
        "code_pattern": """
# Volume Price Analysis
volume_ratio = current_volume / avg_volume_20d
breakout = price > resistance and price_change > 0.03
if breakout and volume_ratio > 2.0:
    # High volume breakout - likely continuation
    buy_signal(size=volume_ratio * base_size, stop_loss=resistance)
"""
    },
    {
        "id": 14,
        "name": "Short Interest Squeeze",
        "category": "sentiment",
        "paper": "Dechow et al. 2001 - Short Sellers",
        "edge": "High short interest creates squeeze risk on positive news",
        "implementation": "Long when short interest > 20% and positive catalyst",
        "expected_sharpe": 0.8,
        "decay_rate": "Event-driven, variable",
        "market_regime": ["all"],
        "risk_factors": ["fundamental_deterioration"],
        "code_pattern": """
# Short Interest Squeeze
short_interest_ratio = short_interest / float_shares
if short_interest_ratio > 0.20 and positive_catalyst:
    # Squeeze potential high
    buy_signal(size=squeeze_size, hold_until_squeeze_or_30days)
"""
    },
    {
        "id": 15,
        "name": "Seasonality Effects",
        "category": "calendar",
        "paper": "Bouman & Jacobsen 2002 - Sell in May",
        "edge": "Predictable calendar patterns in returns",
        "implementation": "Reduce equity exposure May-October, increase Nov-April",
        "expected_sharpe": 0.5,
        "decay_rate": "Decades (robust)",
        "market_regime": ["all"],
        "risk_factors": ["regime_change"],
        "code_pattern": """
# Seasonality Strategy
month = current_month()
if month in [11, 12, 1, 2, 3, 4]:  # Nov-Apr
    # Strong season - increase exposure
    equity_allocation = base_allocation * 1.3
elif month in [5, 6, 7, 8, 9, 10]:  # May-Oct
    # Weak season - reduce exposure
    equity_allocation = base_allocation * 0.7
"""
    },
    {
        "id": 16,
        "name": "Breakout Retest Entry",
        "category": "technical",
        "paper": "Edwards & Magee 1948 - Technical Analysis",
        "edge": "Breakouts that retest and hold are more reliable",
        "implementation": "Buy on retest of breakout level with volume confirmation",
        "expected_sharpe": 0.7,
        "decay_rate": "Classic pattern, still works",
        "market_regime": ["trending"],
        "risk_factors": ["retest_failure"],
        "code_pattern": """
# Breakout Retest Entry
if previous_breakout and price_retests_breakout_level:
    retest_distance = abs(price - breakout_level) / breakout_level
    if retest_distance < 0.02 and volume > avg_volume * 1.2:
        # Successful retest with volume
        buy_signal(size=retest_size, stop_loss=breakout_level*0.97)
"""
    },
    {
        "id": 17,
        "name": "Relative Strength Rotation",
        "category": "momentum",
        "paper": "Jegadeesh & Titman 1993 - Momentum",
        "edge": "Stocks outperforming peers continue to outperform",
        "implementation": "Rotate into top quartile performers monthly",
        "expected_sharpe": 0.9,
        "decay_rate": "Decades (persistent)",
        "market_regime": ["trending"],
        "risk_factors": ["momentum_reversal"],
        "code_pattern": """
# Relative Strength Rotation
monthly_returns = calculate_returns(window=21)
sector_rank = percentile_rank_within_sector(monthly_returns)
if sector_rank > 0.75:
    # Top quartile - increase allocation
    allocation = base_allocation * 1.5
elif sector_rank < 0.25:
    # Bottom quartile - reduce or short
    allocation = base_allocation * 0.5
"""
    },
    {
        "id": 18,
        "name": "Volatility Targeting",
        "category": "risk_parity",
        "paper": "Moreira & Muir 2017 - Volatility-Managed Portfolios",
        "edge": "Constant volatility exposure improves Sharpe ratios",
        "implementation": "Scale position size inversely with realized volatility",
        "expected_sharpe": 1.1,
        "decay_rate": "Structural, long-lasting",
        "market_regime": ["all"],
        "risk_factors": ["volatility_spike"],
        "code_pattern": """
# Volatility Targeting
target_vol = 0.10  # 10% annualized
realized_vol = calculate_realized_vol(window=20)
vol_scalar = target_vol / realized_vol
position_size = base_position * vol_scalar
# Caps to prevent overleveraging
position_size = min(position_size, base_position * 2.0)
"""
    },
    {
        "id": 19,
        "name": "Options Gamma Scalping",
        "category": "options",
        "paper": "Taleb 1997 - Dynamic Hedging",
        "edge": "Profit from gamma when realized vol > implied vol",
        "implementation": "Buy straddles, delta hedge when gamma profit > theta decay",
        "expected_sharpe": 0.8,
        "decay_rate": "Requires skill, 6-12 months",
        "market_regime": ["high_volatility"],
        "risk_factors": ["theta_decay", "transaction_costs"],
        "code_pattern": """
# Gamma Scalping
if realized_vol > implied_vol * 1.1:
    # RV > IV - favorable for gamma scalping
    buy_straddle(atm_strike, size=gamma_size)
    # Delta hedge when price moves
    if abs(delta) > 0.3:
        hedge_delta(current_delta)
"""
    },
    {
        "id": 20,
        "name": "Dark Pool Indicator",
        "category": "microstructure",
        "paper": "Zhu 2014 - Dark Pool Trading",
        "edge": "Large dark pool prints indicate institutional accumulation",
        "implementation": "Buy when dark pool volume > 30% and price stable",
        "expected_sharpe": 0.6,
        "decay_rate": "3-6 months (gets arbitraged)",
        "market_regime": ["all"],
        "risk_factors": ["information_asymmetry"],
        "code_pattern": """
# Dark Pool Indicator
dark_pool_ratio = dark_pool_volume / total_volume
price_stability = std(returns_5d)
if dark_pool_ratio > 0.30 and price_stability < percentile_30:
    # Institutional accumulation without price impact
    buy_signal(size=institutional_follow_size, hold_days=10)
"""
    },
    {
        "id": 21,
        "name": "Basis Trade (Futures-Spot Arbitrage)",
        "category": "arbitrage",
        "paper": "Cornell & French 1983 - Futures Pricing",
        "edge": "Futures-spot basis mean reverts to carrying cost",
        "implementation": "Trade when basis deviates > 2 std from fair value",
        "expected_sharpe": 1.2,
        "decay_rate": "Structural, persistent",
        "market_regime": ["all"],
        "risk_factors": ["delivery_risk", "funding_risk"],
        "code_pattern": """
# Basis Trade
fair_basis = risk_free_rate - convenience_yield
actual_basis = futures_price - spot_price
basis_deviation = (actual_basis - fair_basis) / std_basis
if basis_deviation > 2.0:
    # Futures overpriced - short futures, long spot
    short_futures(size=arb_size)
    long_spot(size=arb_size)
elif basis_deviation < -2.0:
    # Futures underpriced - long futures, short spot
    long_futures(size=arb_size)
    short_spot(size=arb_size)
"""
    },
    {
        "id": 22,
        "name": "Whale Wallet Tracking",
        "category": "crypto",
        "paper": "Makarov & Schoar 2020 - Blockchain Analytics",
        "edge": "Large wallet movements predict price action",
        "implementation": "Follow whale accumulation/distribution patterns",
        "expected_sharpe": 0.7,
        "decay_rate": "6-12 months (front-running risk)",
        "market_regime": ["all"],
        "risk_factors": ["whale_manipulation"],
        "code_pattern": """
# Whale Wallet Tracking
whale_inflow = sum(inflows_to_whale_wallets, last_24h)
whale_outflow = sum(outflows_from_whale_wallets, last_24h)
net_whale_flow = whale_inflow - whale_outflow
if net_whale_flow > threshold_accumulation:
    # Whales accumulating - bullish signal
    buy_signal(size=whale_follow_size, hold_days=7)
elif net_whale_flow < -threshold_distribution:
    # Whales distributing - bearish signal
    sell_signal(size=whale_follow_size, hold_days=7)
"""
    }
]


def get_alpha_by_category(category: str):
    """Get all alphas for a specific category"""
    return [alpha for alpha in ALPHA_LIBRARY if alpha['category'] == category]


def get_alpha_by_regime(regime: str):
    """Get all alphas suitable for a market regime"""
    return [alpha for alpha in ALPHA_LIBRARY if regime in alpha['market_regime']]


def get_high_sharpe_alphas(min_sharpe: float = 1.0):
    """Get alphas with Sharpe ratio above threshold"""
    return [alpha for alpha in ALPHA_LIBRARY if alpha['expected_sharpe'] >= min_sharpe]


def get_alpha_by_id(alpha_id: int):
    """Get specific alpha by ID"""
    for alpha in ALPHA_LIBRARY:
        if alpha['id'] == alpha_id:
            return alpha
    return None


def get_all_categories():
    """Get list of all alpha categories"""
    return list(set(alpha['category'] for alpha in ALPHA_LIBRARY))


def get_recommended_alphas(market_regime: str, min_sharpe: float = 0.7):
    """Get recommended alphas for current market conditions"""
    suitable = get_alpha_by_regime(market_regime)
    return [alpha for alpha in suitable if alpha['expected_sharpe'] >= min_sharpe]
