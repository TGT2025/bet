"""
Persistence Manager - Saves candidate signals and metrics history
Creates artifact table for iteration tracking
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime


class PersistenceManager:
    """Manages persistence of signals and metrics across iterations"""
    
    def __init__(
        self,
        candidates_dir: str = "candidates",
        metrics_file: str = "metrics_history.json"
    ):
        self.candidates_dir = candidates_dir
        self.metrics_file = metrics_file
        os.makedirs(self.candidates_dir, exist_ok=True)
    
    def save_candidate_signals(
        self,
        iteration: int,
        signals_df: pd.DataFrame,
        status: str
    ) -> str:
        """
        Save candidate signals for an iteration
        
        Args:
            iteration: Iteration number
            signals_df: DataFrame with standardized signal format
            status: One of: viable, provisional, low_signal, interface_error, low_alpha
        
        Returns:
            Path to saved file
        """
        filename = f"iteration_{iteration}_{status}_signals.csv"
        filepath = os.path.join(self.candidates_dir, filename)
        
        try:
            signals_df.to_csv(filepath, index=False)
            return filepath
        except Exception as e:
            print(f"Warning: Failed to save candidate signals: {e}")
            return ""
    
    def save_metrics(
        self,
        iteration: int,
        alpha_score: float,
        metrics: Dict[str, Any],
        status: str,
        failures: List[str] = None
    ):
        """
        Append iteration metrics to history
        
        Args:
            iteration: Iteration number
            alpha_score: Calculated alpha score
            metrics: Dict with trade metrics (total_trades, wins, losses, etc.)
            status: Status tag (viable, provisional, etc.)
            failures: List of failure tags
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "alpha_score": alpha_score,
            "status": status,
            "metrics": metrics,
            "failures": failures or []
        }
        
        # Load existing history
        history = []
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    history = json.load(f)
            except Exception:
                pass
        
        # Append new entry
        history.append(entry)
        
        # Save updated history
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save metrics history: {e}")
    
    def load_metrics_history(self) -> List[Dict[str, Any]]:
        """Load full metrics history"""
        if not os.path.exists(self.metrics_file):
            return []
        
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def get_latest_viable(self) -> Dict[str, Any]:
        """Get most recent viable iteration metrics"""
        history = self.load_metrics_history()
        viable = [entry for entry in history if entry.get('status') == 'viable']
        
        if viable:
            return viable[-1]
        return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics across all iterations"""
        history = self.load_metrics_history()
        
        if not history:
            return {
                "total_iterations": 0,
                "viable_count": 0,
                "avg_alpha": 0.0,
                "max_alpha": 0.0,
                "failure_summary": {}
            }
        
        viable = [e for e in history if e.get('status') == 'viable']
        alphas = [e['alpha_score'] for e in history if 'alpha_score' in e]
        
        # Aggregate failures
        failure_summary = {}
        for entry in history:
            for failure in entry.get('failures', []):
                failure_summary[failure] = failure_summary.get(failure, 0) + 1
        
        return {
            "total_iterations": len(history),
            "viable_count": len(viable),
            "avg_alpha": sum(alphas) / len(alphas) if alphas else 0.0,
            "max_alpha": max(alphas) if alphas else 0.0,
            "failure_summary": failure_summary,
            "recent_status": [e.get('status') for e in history[-10:]]
        }


# For standalone testing
if __name__ == "__main__":
    pm = PersistenceManager("/tmp/test_candidates", "/tmp/test_metrics.json")
    
    # Create sample signals
    signals_data = {
        'timestamp': [1234567890, 1234567900],
        'symbol': ['BTCUSDT', 'ETHUSDT'],
        'side': ['LONG', 'SHORT'],
        'size': [0.5, 0.5],
        'entry_price': [50000.0, 3000.0],
        'stop_loss': [49000.0, 3060.0],
        'take_profit': [51500.0, 2910.0],
        'confidence': [0.7, 0.65],
        'reason': ['momentum', 'reversal']
    }
    signals_df = pd.DataFrame(signals_data)
    
    # Save signals
    pm.save_candidate_signals(1, signals_df, 'viable')
    
    # Save metrics
    pm.save_metrics(
        iteration=1,
        alpha_score=0.523,
        metrics={
            'total_trades': 14,
            'wins': 9,
            'losses': 5,
            'diversity': 0.5,
            'trade_ratio': 0.28
        },
        status='viable',
        failures=[]
    )
    
    print("Statistics:", json.dumps(pm.get_statistics(), indent=2))
