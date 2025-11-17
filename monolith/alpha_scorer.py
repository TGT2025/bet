"""
Alpha Scorer - Dynamic alpha scoring for strategy evaluation
Implements adaptive threshold logic
"""

from typing import Dict, Any


class AlphaScorer:
    """Calculates dynamic alpha scores for strategy evaluation"""
    
    def __init__(
        self,
        base_threshold: float = 0.35,  # LOWERED FOR AGGRESSIVE TRADING
        lowered_threshold: float = 0.25,  # LOWERED FOR AGGRESSIVE TRADING
        weights: Dict[str, float] = None
    ):
        self.base_threshold = base_threshold
        self.lowered_threshold = lowered_threshold
        
        # Default weights from spec
        self.weights = weights or {
            'diversity': 0.35,
            'avg_confidence': 0.40,
            'trade_ratio': 0.25
        }
    
    def calculate_alpha(
        self,
        diversity: float,
        avg_confidence: float,
        trade_ratio: float
    ) -> float:
        """
        Calculate alpha score using weighted formula
        
        Formula: alpha = 0.35*diversity + 0.40*avg_confidence + 0.25*trade_ratio
        
        Args:
            diversity: Unique symbols / total possible symbols (0-1)
            avg_confidence: Mean confidence of all signals (0-1)
            trade_ratio: min(1, total_trades / trade_cap) where trade_cap = 50
        
        Returns:
            Alpha score (0-1)
        """
        alpha = (
            self.weights['diversity'] * diversity +
            self.weights['avg_confidence'] * avg_confidence +
            self.weights['trade_ratio'] * trade_ratio
        )
        
        return min(1.0, max(0.0, alpha))
    
    def calculate_metrics_from_signals(
        self,
        signals_df,
        total_symbols: int = 10,
        trade_cap: int = 50
    ) -> Dict[str, float]:
        """
        Calculate alpha metrics from signals DataFrame
        
        Args:
            signals_df: DataFrame with standardized signal format
            total_symbols: Total number of possible symbols
            trade_cap: Trade capacity for ratio calculation
        
        Returns:
            Dict with diversity, avg_confidence, trade_ratio, alpha_score
        """
        if signals_df is None or len(signals_df) == 0:
            return {
                'diversity': 0.0,
                'avg_confidence': 0.0,
                'trade_ratio': 0.0,
                'alpha_score': 0.0,
                'total_trades': 0,
                'unique_symbols': 0
            }
        
        # Calculate diversity
        unique_symbols = signals_df['symbol'].nunique()
        diversity = unique_symbols / max(1, total_symbols)
        
        # Calculate average confidence
        if 'confidence' in signals_df.columns:
            avg_confidence = signals_df['confidence'].mean()
        else:
            avg_confidence = 0.6  # Default
        
        # Calculate trade ratio
        total_trades = len(signals_df)
        trade_ratio = min(1.0, total_trades / trade_cap)
        
        # Calculate alpha score
        alpha_score = self.calculate_alpha(diversity, avg_confidence, trade_ratio)
        
        return {
            'diversity': diversity,
            'avg_confidence': avg_confidence,
            'trade_ratio': trade_ratio,
            'alpha_score': alpha_score,
            'total_trades': total_trades,
            'unique_symbols': unique_symbols
        }
    
    def get_threshold(self, should_lower: bool = False) -> float:
        """
        Get current alpha threshold
        
        Args:
            should_lower: Whether to use lowered threshold (based on failure history)
        
        Returns:
            Threshold value
        """
        return self.lowered_threshold if should_lower else self.base_threshold
    
    def classify_status(
        self,
        alpha_score: float,
        diversity_gate_passed: bool,
        should_lower_threshold: bool = False
    ) -> str:
        """
        Classify iteration status based on alpha score and diversity gate
        
        Returns:
            One of: 'viable', 'low_alpha', 'low_signal'
        """
        if not diversity_gate_passed:
            return 'low_signal'
        
        threshold = self.get_threshold(should_lower_threshold)
        
        if alpha_score >= threshold:
            return 'viable'
        else:
            return 'low_alpha'


# For standalone testing
if __name__ == "__main__":
    import pandas as pd
    
    scorer = AlphaScorer()
    
    # Test with sample signals
    signals_data = {
        'symbol': ['BTCUSDT', 'ETHUSDT', 'BTCUSDT', 'ADAUSDT', 'LTCUSDT'],
        'confidence': [0.7, 0.65, 0.75, 0.6, 0.68]
    }
    signals_df = pd.DataFrame(signals_data)
    
    metrics = scorer.calculate_metrics_from_signals(signals_df, total_symbols=10, trade_cap=50)
    
    print("Metrics:", metrics)
    print(f"Alpha Score: {metrics['alpha_score']:.3f}")
    print(f"Status (normal threshold): {scorer.classify_status(metrics['alpha_score'], True, False)}")
    print(f"Status (lowered threshold): {scorer.classify_status(metrics['alpha_score'], True, True)}")
    
    # Test thresholds
    print(f"\nBase threshold: {scorer.base_threshold}")
    print(f"Lowered threshold: {scorer.lowered_threshold}")
    print(f"Current threshold (normal): {scorer.get_threshold(False)}")
    print(f"Current threshold (lowered): {scorer.get_threshold(True)}")
