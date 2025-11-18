"""
MONOLITH MODULE - Trading System Interface Contracts

This module defines the interface contracts and specifications for the trading system.

INTERFACE CONTRACTS

1. Market Data:
   Type: Dict[str, pd.DataFrame]
   Required columns: ['timestamp','open','high','low','close','volume']
   If 'high' or 'low' missing:
     high = close * (1 + 0.001)
     low  = close * (1 - 0.001)
   Normalization:
     - Forward fill then drop remaining NaNs
     - Ascending by timestamp
   Minimum rows per symbol: 30

2. Strategy Output:
   generate_signals(market_data: Dict[str, pd.DataFrame]) -> List[Dict] (existing monolith)
   Each dict should minimally include:
     { 'symbol': str, 'action': 'BUY'|'SELL', 'price': float, 'strategy': str, optional 'confidence': float }
   ADAPTER will convert these to standardized DataFrame for scoring & persistence:
     ['timestamp','symbol','side','size','entry_price','stop_loss','take_profit','confidence','reason']
   If list empty for normal OHLCV noise, regeneration triggers.

3. Orchestrator:
   AlphaOrchestrator.get_consensus_signal(market_data) -> Dict:
     direction: int (-1,0,1)
     confidence: float (0..1)
     institutional_flow: float (0..1)
     reasoning: str
   No markdown fences, no example usage.

4. Backtester (local evaluation component for adaptive engine phase):
   run(market_data: Dict[str,pd.DataFrame], signals_df: pd.DataFrame) -> Dict metrics:
     total_trades, wins, losses, gross_pnl, net_pnl, avg_confidence, diversity, trade_ratio
   Must normalize market data as per section 1.

5. Alpha Scoring Formula:
   alpha = 0.35*diversity + 0.40*avg_confidence + 0.25*trade_ratio
   Base threshold = 0.35 (LOWERED FOR AGGRESSIVE TRADING)
   If cumulative (LOW_SIGNAL + INTERFACE_ERROR) >= 3 -> lower threshold to 0.25 until one viable iteration passes.

6. Diversity Gate (LOOSENED FOR MORE TRADES):
   Pass if (signals >= 3) OR (>=1 symbol AND >=2 signals)

7. Error Taxonomy Tags:
   SYNTAX_ERROR, INTERFACE_ERROR, LOW_SIGNAL, LOW_ALPHA, RUNTIME_FAIL, BACKTEST_FALLBACK, NORMALIZATION_WARN

8. Fallback:
   Orchestrator neutral (direction=0, confidence<0.1) -> strategy may emit technical fallback signals (still list-of-dicts, adapter handles DataFrame format).

9. Persistence:
   Save candidate signals (converted DataFrame) every iteration even on failure into:
     candidates/iteration_<n>_{viable|provisional|low_signal|interface_error|low_alpha}_signals.csv
   Append metrics + alpha in metrics_history.json.

10. Prohibited in generated files:
   - Markdown fences (```).
   - Example usage blocks in orchestrator/strategies.
   - Random synthetic fake OHLCV generation inside core trading logic (except your controlled testing utilities).
   - Removing AdaptiveTradingStrategy class.
   - Placeholder implementations or TODO comments in production code.

11. SPECIALIZED QUANT AGENT ARCHITECTURE (REQUIRED):

   A. ResearchQuantAgent - Provides academic alpha sources and market intelligence
      Must implement:
      - get_alpha_sources() -> List[Dict] with fields:
        * name: str (alpha source name)
        * edge: str (exploitable inefficiency)
        * implementation: str (concrete trading rule)
        * expected_sharpe: float (backtested Sharpe ratio)
        * decay_rate: str (how long edge persists)

      - analyze_market_regime() -> Dict with:
        * current_regime: str ('low_vol', 'high_vol', 'trending', 'mean_reverting')
        * recommended_strategies: List[str]
        * avoid_strategies: List[str]

      - fetch_online_research() -> List[Dict] (OPTIONAL - can fetch from academic sources):
        * Can use LLM to read and reason about academic papers
        * Can fetch from arXiv, SSRN, QuantResearch websites
        * Can analyze recent trading research and adapt
        * Fallback to built-in library if offline

      Built-in alpha library (monolith/alpha_library.py - 22+ proven sources):
        * Volatility Risk Premium (sell options when IV > RV)
        * Momentum Crash Protection (avoid momentum in rebounds)
        * Liquidity Provision (market making at bid/ask)
        * Statistical Arbitrage (cointegration pairs)
        * Cross-Asset Correlation Breakdowns
        * Funding Rate Arbitrage (crypto perpetuals)
        * Order Flow Imbalance
        * Trend Following with Regime Filter
        * Mean Reversion at Support/Resistance
        * Carry Trade with Risk Management
        * Earnings Drift Post-Announcement
        * Overnight Gap Fade
        * Volume Price Analysis
        * Short Interest Squeeze
        * Seasonality Effects
        * Breakout Retest Entry
        * Relative Strength Rotation
        * Volatility Targeting
        * Options Gamma Scalping
        * Dark Pool Indicator
        * Basis Trade (Futures-Spot Arbitrage)
        * Whale Wallet Tracking (crypto)

      Usage: from monolith.alpha_library import ALPHA_LIBRARY, get_alpha_by_regime

   B. ExecutionQuantAgent - Optimizes trade execution quality
      Must implement:
      - optimize_entry(strategy_code: str) -> str
        Adds: Limit orders, time-weighted scheduling, liquidity seeking

      - optimize_exit(strategy_code: str) -> str
        Adds: ATR-based trailing stops, volatility-adjusted targets

      - reduce_slippage(strategy_code: str) -> str
        Adds: Volume-weighted execution, avoid high-vol periods

      - minimize_costs(strategy_code: str) -> str
        Adds: Batch orders, maker fee rebates, profit thresholds

   C. RiskQuantAgent - Manages portfolio risk and position sizing
      Must implement:
      - calculate_position_sizes(performance: Dict) -> Dict with:
        * kelly_size: float (Kelly criterion allocation)
        * risk_parity_size: float (risk-balanced allocation)
        * vol_target_size: float (volatility targeting)

      - validate_risk_limits(performance: Dict) -> bool
        Checks: max_drawdown <= 20%, VaR_95 <= 3%, concentration <= 25%

      - generate_stress_scenarios() -> Dict
        Returns: flash_crash, vol_spike, liquidity_crisis scenarios

12. QUANT AGENT INTEGRATION FLOW:

    Phase 0 (NEW): Research Agent provides alpha sources
    Phase 1: Reasoner creates plan using researched alphas (not random ideas)
    Phase 2: Coder implements strategies based on proven edges
    Phase 2.5 (NEW): Execution Agent optimizes entry/exit/slippage
    Phase 3: Enforcer tests with realistic constraints
    Phase 3.5 (NEW): Risk Agent validates position sizing and limits
    Phase 4: Champion promotion only if risk-approved

13. AGENT COMMUNICATION & PERSISTENT MEMORY (CRITICAL):

    ALL agents MUST communicate with each other and create persistent memory:

    A. Agent Communication Protocol:
       - ResearchQuant -> Reasoner: Provides alpha sources and market regime
       - Reasoner -> Coder: Passes implementation plan with alpha specifications
       - Coder -> ExecutionQuant: Sends raw strategy code for optimization
       - ExecutionQuant -> Coder: Returns optimized code
       - Coder -> RiskQuant: Sends strategy for risk validation
       - RiskQuant -> Reasoner: Provides risk limits and position sizing
       - ALL agents -> AgentActivityTracker: Log thoughts, suggestions, decisions

    B. Persistent Memory Creation:
       Each agent MUST maintain state across iterations:

       - ResearchQuant memory:
         * Previously tested alpha sources (success/failure rates)
         * Market regime history (what worked when)
         * Alpha decay tracking (which alphas are getting crowded)

       - ExecutionQuant memory:
         * Optimization effectiveness (slippage reduction achieved)
         * Best entry/exit patterns per market regime
         * Failed optimization attempts (to avoid repeating)

       - RiskQuant memory:
         * Historical risk metrics per strategy type
         * Stress test results archive
         * Position sizing effectiveness (Kelly vs VaR vs VolTarget)

       - Reasoner memory:
         * Decision outcomes (which plans led to viable strategies)
         * Failed reasoning patterns
         * Successful alpha combinations

       - Coder memory:
         * Code patterns that passed validation
         * Common syntax errors to avoid
         * Implementation success rates

    C. Memory Storage Implementation:
       All agents use AgentActivityTracker for persistence:

       Example usage:
       
       # Log agent thinking
       activity_tracker.log_agent_thinking(
           agent_name="ResearchQuant",
           thought="Analyzing volatility risk premium for current regime",
           context={"market_regime": "high_volatility", "vix": 28.5}
       )

       # Log agent suggestion
       activity_tracker.log_agent_suggestion(
           agent_name="ExecutionQuant",
           suggestion_type="entry_optimization",
           suggestion={"method": "limit_orders", "expected_slippage_reduction": 0.0025}
       )

       # Log strategy details
       activity_tracker.log_strategy_info(
           iteration=5,
           strategy_data={
               "alpha_sources_used": [...],
               "execution_optimizations": [...],
               "risk_parameters": {...}
           }
       )

    D. Inter-Agent Data Flow:
       Agents share data via context dictionaries:

       Example structures:
       
       # ResearchQuant provides to Reasoner
       research_context = {
           "alpha_sources": get_alpha_by_regime(current_regime),
           "market_regime": "high_volatility",
           "recommended_alphas": ["volatility_risk_premium", "gamma_scalping"]
       }

       # Reasoner provides to Coder
       implementation_plan = {
           "alpha_to_implement": "volatility_risk_premium",
           "research_context": research_context,
           "execution_requirements": ["limit_orders", "delta_hedging"]
       }

       # ExecutionQuant provides to Coder
       optimized_code = {
           "original_code": strategy_code,
           "optimizations_applied": ["limit_orders", "atr_stops"],
           "expected_improvement": {"slippage_reduction": 0.0025}
       }

       # RiskQuant provides to all
       risk_framework = {
           "position_sizes": {"kelly": 0.15, "vol_target": 0.18},
           "risk_limits": {"max_dd": 0.15, "var_95": 0.02},
           "validation_status": "approved"
       }

14. AGENT DATA SOURCES (NO PLACEHOLDERS):

    ResearchQuantAgent data sources:
    - Built-in alpha library: monolith/alpha_library.py (22+ proven sources with code patterns)
    - Market regime detection from price/volume patterns (self-contained calculations)
    - OPTIONAL online fetch: LLM can read academic papers from arXiv, SSRN, QuantResearch
      * Use LLM reasoning to extract alpha sources from papers
      * Parse trading research PDFs and adapt strategies
      * Fallback to built-in library if network unavailable
    - No external API calls required for basic operation

    ExecutionQuantAgent optimizations:
    - Pattern-based code transformations (no AI needed)
    - Deterministic optimization rules (hardcoded best practices)
    - No external dependencies
    - Slippage models based on market microstructure research

    RiskQuantAgent calculations:
    - Mathematical formulas (Kelly, VaR, Sharpe, Sortino)
    - Performance-based position sizing algorithms
    - No external risk feeds needed
    - Stress test scenarios hardcoded (flash crash, vol spike, liquidity crisis)
"""

# Version information
__version__ = "1.0.0"
__all__ = []

# This module serves as documentation and interface specification
# Actual implementation classes should be imported from submodules
