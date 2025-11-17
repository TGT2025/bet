"""
Memory Manager - Persistent training memory across iterations
Tracks failures, alpha scores, and viable strategies
"""

import json
import os
from typing import Dict, List, Any
from datetime import datetime


class MemoryManager:
    """Manages persistent memory state for adaptive learning"""
    
    def __init__(self, memory_file: str = "training_memory.json"):
        self.memory_file = memory_file
        self.state = self._load_or_initialize()
    
    def _load_or_initialize(self) -> Dict[str, Any]:
        """Load existing memory or initialize new state"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Initialize fresh memory state
        return {
            "iterations": 0,
            "failure_counts": {},
            "recent_failures": [],
            "alpha_history": [],
            "viable_strategies": 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def update_iteration(self, iteration_data: Dict[str, Any]):
        """
        Update memory with iteration results
        
        Args:
            iteration_data: Dict with keys: failures, alpha_score, metrics, status
        """
        self.state["iterations"] += 1
        self.state["last_updated"] = datetime.now().isoformat()
        
        # Update failure counts
        failures = iteration_data.get('failures', [])
        for failure_tag in failures:
            self.state["failure_counts"][failure_tag] = \
                self.state["failure_counts"].get(failure_tag, 0) + 1
        
        # Keep recent failures (last 20)
        if failures:
            self.state["recent_failures"].extend(failures)
            self.state["recent_failures"] = self.state["recent_failures"][-20:]
        
        # Record alpha history
        alpha_score = iteration_data.get('alpha_score')
        metrics = iteration_data.get('metrics', {})
        status = iteration_data.get('status', 'unknown')
        
        if alpha_score is not None:
            self.state["alpha_history"].append({
                "timestamp": time.time(),
                "iteration": self.state["iterations"],
                "alpha_score": alpha_score,
                "status": status,
                "metrics": metrics
            })
            
            # Keep last 50 alpha scores
            if len(self.state["alpha_history"]) > 50:
                self.state["alpha_history"] = self.state["alpha_history"][-50:]
        
        # Track viable strategies
        if status == 'viable':
            self.state["viable_strategies"] += 1
        
        self._save()
    
    def get_failure_count(self, tag: str) -> int:
        """Get count for specific failure tag"""
        return self.state["failure_counts"].get(tag, 0)
    
    def get_recent_failures(self, limit: int = 10) -> List[str]:
        """Get most recent failures"""
        return self.state["recent_failures"][-limit:]
    
    def get_alpha_stats(self) -> Dict[str, float]:
        """Calculate alpha statistics"""
        if not self.state["alpha_history"]:
            return {
                "mean": 0.0,
                "max": 0.0,
                "recent_mean": 0.0,
                "trend": 0.0
            }
        
        scores = [entry["alpha_score"] for entry in self.state["alpha_history"]]
        recent_scores = scores[-10:] if len(scores) >= 10 else scores
        
        stats = {
            "mean": sum(scores) / len(scores),
            "max": max(scores),
            "recent_mean": sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
        }
        
        # Calculate trend (simple slope of recent scores)
        if len(recent_scores) >= 3:
            n = len(recent_scores)
            x_mean = (n - 1) / 2
            y_mean = stats["recent_mean"]
            numerator = sum((i - x_mean) * (y - y_mean) 
                          for i, y in enumerate(recent_scores))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            stats["trend"] = numerator / denominator if denominator != 0 else 0.0
        else:
            stats["trend"] = 0.0
        
        return stats
    
    def should_lower_threshold(self) -> bool:
        """Determine if alpha threshold should be temporarily lowered"""
        low_signal = self.get_failure_count('LOW_SIGNAL')
        interface_error = self.get_failure_count('INTERFACE_ERROR')
        return (low_signal + interface_error) >= 6
    
    def _save(self):
        """Save memory state to disk"""
        try:
            os.makedirs(os.path.dirname(self.memory_file) if os.path.dirname(self.memory_file) else ".", 
                       exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save memory state: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current memory state"""
        return self.state.copy()


# For standalone testing
if __name__ == "__main__":
    import time
    
    # Test memory manager
    mm = MemoryManager("/tmp/test_memory.json")
    
    # Simulate some iterations
    mm.update_iteration({
        'failures': ['SYNTAX_ERROR'],
        'alpha_score': 0.45,
        'metrics': {'total_trades': 10},
        'status': 'provisional'
    })
    
    mm.update_iteration({
        'failures': ['LOW_SIGNAL'],
        'alpha_score': 0.52,
        'metrics': {'total_trades': 15},
        'status': 'viable'
    })
    
    print("Memory state:", json.dumps(mm.get_state(), indent=2))
    print("Alpha stats:", mm.get_alpha_stats())
    print("Should lower threshold:", mm.should_lower_threshold())
