"""
Signal Adapter - Converts list-of-dict signals to DataFrame format
Maintains compatibility with existing list-of-dict strategy outputs
"""

import pandas as pd
import time
from typing import List, Dict, Any


class SignalAdapter:
    """Adapter to convert list-of-dict signals to standardized DataFrame format"""
    
    def __init__(self):
        self.required_keys = ['symbol', 'action', 'price', 'strategy']
        self.columns = [
            'timestamp', 'symbol', 'side', 'size', 'entry_price',
            'stop_loss', 'take_profit', 'confidence', 'reason'
        ]
    
    def signals_to_dataframe(self, signals: List[Dict[str, Any]], market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Convert list of signal dicts to standardized DataFrame
        
        Args:
            signals: List of dicts with keys: symbol, action, price, strategy, [confidence]
            market_data: Dict mapping symbols to their OHLCV DataFrames
        
        Returns:
            DataFrame with columns: timestamp, symbol, side, size, entry_price,
                                   stop_loss, take_profit, confidence, reason
        """
        if not signals:
            return pd.DataFrame(columns=self.columns)
        
        rows = []
        for sig in signals:
            try:
                # Validate required keys
                if not all(key in sig for key in self.required_keys):
                    continue
                
                symbol = sig['symbol']
                action = sig['action']
                price = float(sig['price'])
                
                # Get timestamp from market data or use current time
                df = market_data.get(symbol)
                if df is not None and 'timestamp' in df.columns and len(df) > 0:
                    ts = df.iloc[-1]['timestamp']
                else:
                    ts = int(time.time())
                
                # Convert BUY/SELL to LONG/SHORT
                side = 'LONG' if action.upper() == 'BUY' else 'SHORT'
                
                # Default position size
                size = 0.5
                
                # Calculate stop loss and take profit
                if side == 'LONG':
                    stop_loss = price * 0.98  # 2% stop loss
                    take_profit = price * 1.03  # 3% take profit
                else:
                    stop_loss = price * 1.02  # 2% stop loss for short
                    take_profit = price * 0.97  # 3% take profit for short
                
                # Extract confidence or use default
                confidence = float(sig.get('confidence', 0.6))
                
                # Use strategy name as reason
                reason = sig.get('strategy', 'generated')
                
                rows.append([
                    ts, symbol, side, size, price,
                    stop_loss, take_profit, confidence, reason
                ])
                
            except Exception as e:
                # Skip malformed signals
                continue
        
        if not rows:
            return pd.DataFrame(columns=self.columns)
        
        return pd.DataFrame(rows, columns=self.columns)
    
    def validate_convertible(self, signals: List[Dict[str, Any]]) -> tuple[bool, str]:
        """
        Validate that signals can be converted to DataFrame
        
        Returns:
            (success, error_message)
        """
        if not signals:
            return True, ""
        
        for i, sig in enumerate(signals):
            if not isinstance(sig, dict):
                return False, f"Signal {i} is not a dict"
            
            missing_keys = [key for key in self.required_keys if key not in sig]
            if missing_keys:
                return False, f"Signal {i} missing keys: {missing_keys}"
            
            # Validate types
            try:
                float(sig['price'])
            except (ValueError, TypeError):
                return False, f"Signal {i} has invalid price: {sig.get('price')}"
            
            if sig['action'].upper() not in ['BUY', 'SELL']:
                return False, f"Signal {i} has invalid action: {sig.get('action')}"
        
        return True, ""
