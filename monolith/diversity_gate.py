"""
Diversity Gate - Validates signal diversity requirements
Ensures sufficient signal coverage before proceeding
"""

from typing import List, Dict, Any
import pandas as pd


class DiversityGate:
    """Validates diversity requirements for generated signals"""
    
    def __init__(
        self,
        min_total_signals: int = 6,
        min_symbols: int = 2,
        min_signals_per_symbol: int = 3
    ):
        self.min_total_signals = min_total_signals
        self.min_symbols = min_symbols
        self.min_signals_per_symbol = min_signals_per_symbol
    
    def check_gate(self, signals: List[Dict[str, Any]]) -> tuple[bool, str]:
        """
        Check if signals pass diversity gate
        
        Criteria:
        - Pass if len(signals) >= min_total_signals (default 6)
        - OR if num_symbols >= min_symbols (default 2) AND 
             each symbol has >= min_signals_per_symbol (default 3)
        
        Args:
            signals: List of signal dicts
        
        Returns:
            (passed, reason)
        """
        if not signals:
            return False, "No signals generated"
        
        # Criterion 1: Total signals
        if len(signals) >= self.min_total_signals:
            return True, f"Passed: {len(signals)} >= {self.min_total_signals} total signals"
        
        # Criterion 2: Symbol diversity
        symbol_counts = {}
        for sig in signals:
            symbol = sig.get('symbol')
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        num_symbols = len(symbol_counts)
        
        if num_symbols < self.min_symbols:
            return False, f"Failed: {num_symbols} symbols < {self.min_symbols} required"
        
        # Check if each symbol has enough signals
        symbols_with_enough = [
            sym for sym, count in symbol_counts.items()
            if count >= self.min_signals_per_symbol
        ]
        
        if len(symbols_with_enough) >= self.min_symbols:
            return True, (
                f"Passed: {len(symbols_with_enough)} symbols with "
                f">={self.min_signals_per_symbol} signals each"
            )
        
        return False, (
            f"Failed: Only {len(symbols_with_enough)} symbols have "
            f">={self.min_signals_per_symbol} signals (need {self.min_symbols})"
        )
    
    def check_gate_df(self, signals_df: pd.DataFrame) -> tuple[bool, str]:
        """
        Check if signals DataFrame passes diversity gate
        
        Args:
            signals_df: DataFrame with standardized signal format
        
        Returns:
            (passed, reason)
        """
        if signals_df is None or len(signals_df) == 0:
            return False, "No signals in DataFrame"
        
        # Criterion 1: Total signals
        total_signals = len(signals_df)
        if total_signals >= self.min_total_signals:
            return True, f"Passed: {total_signals} >= {self.min_total_signals} total signals"
        
        # Criterion 2: Symbol diversity
        if 'symbol' not in signals_df.columns:
            return False, "Missing 'symbol' column in DataFrame"
        
        symbol_counts = signals_df['symbol'].value_counts()
        num_symbols = len(symbol_counts)
        
        if num_symbols < self.min_symbols:
            return False, f"Failed: {num_symbols} symbols < {self.min_symbols} required"
        
        # Check if each symbol has enough signals
        symbols_with_enough = symbol_counts[symbol_counts >= self.min_signals_per_symbol]
        
        if len(symbols_with_enough) >= self.min_symbols:
            return True, (
                f"Passed: {len(symbols_with_enough)} symbols with "
                f">={self.min_signals_per_symbol} signals each"
            )
        
        return False, (
            f"Failed: Only {len(symbols_with_enough)} symbols have "
            f">={self.min_signals_per_symbol} signals (need {self.min_symbols})"
        )
    
    def get_diversity_metrics(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate diversity metrics for signals
        
        Returns:
            Dict with total_signals, num_symbols, signals_per_symbol, passed
        """
        if not signals:
            return {
                'total_signals': 0,
                'num_symbols': 0,
                'signals_per_symbol': {},
                'passed': False,
                'reason': 'No signals'
            }
        
        symbol_counts = {}
        for sig in signals:
            symbol = sig.get('symbol')
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        passed, reason = self.check_gate(signals)
        
        return {
            'total_signals': len(signals),
            'num_symbols': len(symbol_counts),
            'signals_per_symbol': symbol_counts,
            'passed': passed,
            'reason': reason
        }


# For standalone testing
if __name__ == "__main__":
    gate = DiversityGate()
    
    # Test case 1: Enough total signals
    signals1 = [
        {'symbol': 'BTCUSDT', 'action': 'BUY'},
        {'symbol': 'BTCUSDT', 'action': 'SELL'},
        {'symbol': 'ETHUSDT', 'action': 'BUY'},
        {'symbol': 'ETHUSDT', 'action': 'SELL'},
        {'symbol': 'ADAUSDT', 'action': 'BUY'},
        {'symbol': 'LTCUSDT', 'action': 'BUY'},
    ]
    passed, reason = gate.check_gate(signals1)
    print(f"Test 1: {passed} - {reason}")
    print(f"Metrics: {gate.get_diversity_metrics(signals1)}")
    
    # Test case 2: Good symbol diversity
    signals2 = [
        {'symbol': 'BTCUSDT', 'action': 'BUY'},
        {'symbol': 'BTCUSDT', 'action': 'SELL'},
        {'symbol': 'BTCUSDT', 'action': 'BUY'},
        {'symbol': 'ETHUSDT', 'action': 'BUY'},
        {'symbol': 'ETHUSDT', 'action': 'SELL'},
        {'symbol': 'ETHUSDT', 'action': 'BUY'},
    ]
    passed, reason = gate.check_gate(signals2)
    print(f"\nTest 2: {passed} - {reason}")
    
    # Test case 3: Insufficient signals
    signals3 = [
        {'symbol': 'BTCUSDT', 'action': 'BUY'},
        {'symbol': 'ETHUSDT', 'action': 'SELL'},
    ]
    passed, reason = gate.check_gate(signals3)
    print(f"\nTest 3: {passed} - {reason}")
