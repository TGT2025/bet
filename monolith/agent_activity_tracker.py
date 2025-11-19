"""
Agent Activity Tracker - Core component of Monolith Alpha Engine

This module tracks agent activities, learns from patterns, and provides
adaptive recommendations for the trading bot system.

Implements persistent memory creation for agents as per monolith interface contracts.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import threading


logger = logging.getLogger(__name__)


class ActivityRecord:
    """Represents a single agent activity record"""
    
    def __init__(
        self,
        agent_name: str,
        action: str,
        details: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        self.agent_name = agent_name
        self.action = action
        self.details = details
        self.timestamp = timestamp or datetime.now()
        self.id = f"{agent_name}_{action}_{self.timestamp.timestamp()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert activity record to dictionary"""
        return {
            'id': self.id,
            'agent': self.agent_name,
            'action': self.action,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActivityRecord':
        """Create activity record from dictionary"""
        timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(
            agent_name=data['agent'],
            action=data['action'],
            details=data['details'],
            timestamp=timestamp
        )


class AgentActivityTracker:
    """
    Tracks and analyzes agent activities for adaptive learning
    
    Features:
    - Activity logging and persistence
    - Pattern recognition
    - Performance correlation
    - Adaptive recommendations
    - Agent memory persistence across iterations
    """
    
    def __init__(self, storage_dir: str = "monolith_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.activities: List[ActivityRecord] = []
        self.agent_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_actions': 0,
            'action_counts': defaultdict(int),
            'success_count': 0,
            'failure_count': 0,
            'last_activity': None
        })
        
        # Agent memory stores - one per agent type
        self.agent_memories: Dict[str, Dict[str, Any]] = {
            'ResearchQuant': {
                'tested_alpha_sources': {},  # alpha_name -> {success_count, failure_count, last_used}
                'market_regime_history': [],  # {regime, timestamp, what_worked}
                'alpha_decay_tracking': {}  # alpha_name -> decay_metrics
            },
            'ExecutionQuant': {
                'optimization_effectiveness': {},  # optimization_type -> effectiveness_score
                'entry_exit_patterns': {},  # regime -> {best_entry, best_exit}
                'failed_optimizations': []  # List of failed attempts to avoid
            },
            'RiskQuant': {
                'risk_metrics_history': {},  # strategy_type -> historical_metrics
                'stress_test_results': [],  # Archive of stress tests
                'position_sizing_effectiveness': {}  # method -> effectiveness
            },
            'Reasoner': {
                'decision_outcomes': [],  # {decision, outcome, iteration}
                'failed_reasoning_patterns': [],  # Patterns to avoid
                'successful_alpha_combinations': []  # What worked
            },
            'Coder': {
                'code_patterns_passed': [],  # Successful code patterns
                'common_syntax_errors': {},  # error_type -> count
                'implementation_success_rates': {}  # pattern -> success_rate
            }
        }
        
        self.agent_thinking: List[Dict[str, Any]] = []  # Thoughts log
        self.agent_suggestions: List[Dict[str, Any]] = []  # Suggestions log
        self.strategy_info: List[Dict[str, Any]] = []  # Strategy details log
        
        self.performance_correlations: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Load existing data if available
        self._load_state()
        
        logger.info("✅ Agent Activity Tracker initialized with persistent memory")
    
    def log_agent_thinking(
        self,
        agent_name: str,
        thought: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log agent thinking for persistent memory
        
        As per monolith interface contracts, all agents must log their thoughts
        to create persistent memory across iterations.
        
        Args:
            agent_name: Name of the agent (ResearchQuant, ExecutionQuant, RiskQuant, etc.)
            thought: The thought/reasoning process
            context: Additional context about the thinking
        """
        with self._lock:
            thinking_record = {
                'timestamp': datetime.now().isoformat(),
                'agent': agent_name,
                'thought': thought,
                'context': context or {}
            }
            self.agent_thinking.append(thinking_record)
            
            logger.debug(f"💭 {agent_name}: {thought[:100]}...")
        
        # Track as activity (outside lock to avoid deadlock)
        self.track_activity(
            agent_name=agent_name,
            action='thinking',
            details={'thought': thought, 'context': context}
        )
    
    def log_agent_suggestion(
        self,
        agent_name: str,
        suggestion_type: str,
        suggestion: Dict[str, Any],
        priority: str = 'medium'
    ):
        """
        Log agent suggestion for persistent memory
        
        As per monolith interface contracts, agents communicate via suggestions
        that are logged for inter-agent coordination.
        
        Args:
            agent_name: Name of the agent
            suggestion_type: Type of suggestion (entry_optimization, risk_limit, etc.)
            suggestion: The actual suggestion data
            priority: Priority level (low, medium, high)
        """
        with self._lock:
            suggestion_record = {
                'timestamp': datetime.now().isoformat(),
                'agent': agent_name,
                'type': suggestion_type,
                'suggestion': suggestion,
                'priority': priority
            }
            self.agent_suggestions.append(suggestion_record)
            
            logger.debug(f"💡 {agent_name} suggestion ({suggestion_type})")
        
        # Track as activity (outside lock to avoid deadlock)
        self.track_activity(
            agent_name=agent_name,
            action='suggestion',
            details={'type': suggestion_type, 'suggestion': suggestion, 'priority': priority}
        )
    
    def log_strategy_info(
        self,
        iteration: int,
        strategy_data: Dict[str, Any]
    ):
        """
        Log strategy information for persistent memory
        
        As per monolith interface contracts, strategy details are logged
        to track what was tried and what worked.
        
        Args:
            iteration: Current iteration number
            strategy_data: Strategy details including alpha sources, optimizations, risk parameters
        """
        with self._lock:
            strategy_record = {
                'timestamp': datetime.now().isoformat(),
                'iteration': iteration,
                'data': strategy_data
            }
            self.strategy_info.append(strategy_record)
            
            logger.debug(f"📊 Strategy info logged for iteration {iteration}")
    
    def track_activity(
        self,
        agent_name: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
        success: Optional[bool] = None
    ) -> ActivityRecord:
        """
        Track a new agent activity
        
        Args:
            agent_name: Name of the agent
            action: Action performed
            details: Additional details about the action
            success: Whether the action was successful
        
        Returns:
            Created ActivityRecord
        """
        with self._lock:
            details = details or {}
            if success is not None:
                details['success'] = success
            
            record = ActivityRecord(agent_name, action, details)
            self.activities.append(record)
            
            # Update stats
            stats = self.agent_stats[agent_name]
            stats['total_actions'] += 1
            stats['action_counts'][action] += 1
            stats['last_activity'] = record.timestamp
            
            if success is True:
                stats['success_count'] += 1
            elif success is False:
                stats['failure_count'] += 1
            
            # Auto-save periodically
            if len(self.activities) % 50 == 0:
                self._save_state()
            
            logger.debug(f"📝 Tracked: {agent_name}.{action}")
            return record
    
    def update_agent_memory(
        self,
        agent_name: str,
        memory_key: str,
        memory_value: Any
    ):
        """
        Update persistent memory for an agent
        
        Args:
            agent_name: Name of the agent
            memory_key: Key in the agent's memory dict
            memory_value: Value to store
        """
        with self._lock:
            if agent_name not in self.agent_memories:
                self.agent_memories[agent_name] = {}
            
            self.agent_memories[agent_name][memory_key] = memory_value
            logger.debug(f"💾 Updated memory for {agent_name}: {memory_key}")
    
    def get_agent_memory(
        self,
        agent_name: str,
        memory_key: Optional[str] = None
    ) -> Any:
        """
        Retrieve persistent memory for an agent
        
        Args:
            agent_name: Name of the agent
            memory_key: Specific key to retrieve, or None for all memory
        
        Returns:
            Memory value or entire memory dict
        """
        with self._lock:
            if agent_name not in self.agent_memories:
                return {} if memory_key is None else None
            
            if memory_key is None:
                return self.agent_memories[agent_name]
            
            return self.agent_memories[agent_name].get(memory_key)
    
    def track_performance(
        self,
        agent_name: str,
        metric_value: float,
        metric_name: str = "performance"
    ):
        """Track performance metric for an agent"""
        with self._lock:
            key = f"{agent_name}_{metric_name}"
            self.performance_correlations[key].append(metric_value)
            
            # Keep only recent performance data (last 1000 points)
            if len(self.performance_correlations[key]) > 1000:
                self.performance_correlations[key] = self.performance_correlations[key][-1000:]
    
    def get_agent_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get summary statistics for an agent"""
        with self._lock:
            stats = self.agent_stats[agent_name]
            total = stats['total_actions']
            
            if total == 0:
                return {'agent': agent_name, 'no_activity': True}
            
            success_rate = stats['success_count'] / total if total > 0 else 0.0
            
            return {
                'agent': agent_name,
                'total_actions': total,
                'action_breakdown': dict(stats['action_counts']),
                'success_rate': success_rate,
                'success_count': stats['success_count'],
                'failure_count': stats['failure_count'],
                'last_activity': stats['last_activity'].isoformat() if stats['last_activity'] else None
            }
    
    def get_recent_activities(
        self,
        agent_name: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[ActivityRecord]:
        """
        Get recent activities with optional filtering
        
        Args:
            agent_name: Filter by agent name
            action: Filter by action type
            limit: Maximum number of records to return
            since: Only return activities after this time
        
        Returns:
            List of ActivityRecord objects
        """
        with self._lock:
            filtered = self.activities
            
            if agent_name:
                filtered = [r for r in filtered if r.agent_name == agent_name]
            
            if action:
                filtered = [r for r in filtered if r.action == action]
            
            if since:
                filtered = [r for r in filtered if r.timestamp >= since]
            
            # Return most recent first
            return sorted(filtered, key=lambda r: r.timestamp, reverse=True)[:limit]
    
    def get_suggestions_for_agent(
        self,
        target_agent: str,
        suggestion_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get suggestions relevant to a specific agent
        
        Args:
            target_agent: Agent name to get suggestions for
            suggestion_type: Filter by suggestion type
            limit: Maximum number of suggestions
        
        Returns:
            List of suggestion records
        """
        with self._lock:
            filtered = self.agent_suggestions
            
            if suggestion_type:
                filtered = [s for s in filtered if s['type'] == suggestion_type]
            
            # Return most recent first
            return sorted(filtered, key=lambda s: s['timestamp'], reverse=True)[:limit]
    
    def get_activity_patterns(
        self,
        agent_name: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze activity patterns for an agent
        
        Args:
            agent_name: Name of the agent
            time_window_hours: Time window to analyze
        
        Returns:
            Dictionary with pattern analysis
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            recent = [
                r for r in self.activities
                if r.agent_name == agent_name and r.timestamp >= cutoff
            ]
            
            if not recent:
                return {'agent': agent_name, 'no_recent_activity': True}
            
            # Analyze patterns
            action_frequency = defaultdict(int)
            hourly_distribution = defaultdict(int)
            
            for record in recent:
                action_frequency[record.action] += 1
                hour = record.timestamp.hour
                hourly_distribution[hour] += 1
            
            return {
                'agent': agent_name,
                'time_window_hours': time_window_hours,
                'total_activities': len(recent),
                'action_frequency': dict(action_frequency),
                'hourly_distribution': dict(hourly_distribution),
                'most_common_action': max(action_frequency.items(), key=lambda x: x[1])[0] if action_frequency else None
            }
    
    def get_adaptive_recommendations(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate adaptive recommendations based on tracked activities
        
        Args:
            context: Optional context information
        
        Returns:
            List of recommendations
        """
        recommendations = []
        context = context or {}
        
        with self._lock:
            # Analyze all agents
            for agent_name, stats in self.agent_stats.items():
                total = stats['total_actions']
                if total == 0:
                    continue
                
                success_count = stats['success_count']
                failure_count = stats['failure_count']
                
                # Calculate success rate
                if success_count + failure_count > 0:
                    success_rate = success_count / (success_count + failure_count)
                    
                    # Low success rate recommendation
                    if success_rate < 0.5 and total >= 10:
                        recommendations.append({
                            'type': 'agent_performance',
                            'priority': 'high',
                            'agent': agent_name,
                            'message': f"Agent {agent_name} has low success rate ({success_rate:.1%})",
                            'suggestion': 'Consider reviewing agent parameters or strategy'
                        })
                    
                    # High success rate acknowledgment
                    elif success_rate > 0.8 and total >= 20:
                        recommendations.append({
                            'type': 'agent_performance',
                            'priority': 'info',
                            'agent': agent_name,
                            'message': f"Agent {agent_name} performing well ({success_rate:.1%})",
                            'suggestion': 'Consider increasing agent utilization'
                        })
            
            # Check for inactive agents
            now = datetime.now()
            for agent_name, stats in self.agent_stats.items():
                last_activity = stats['last_activity']
                if last_activity:
                    hours_since = (now - last_activity).total_seconds() / 3600
                    if hours_since > 24 and stats['total_actions'] > 0:
                        recommendations.append({
                            'type': 'agent_inactivity',
                            'priority': 'medium',
                            'agent': agent_name,
                            'message': f"Agent {agent_name} inactive for {hours_since:.1f} hours",
                            'suggestion': 'Verify agent is still needed or restart if required'
                        })
        
        return recommendations
    
    def get_performance_trend(
        self,
        agent_name: str,
        metric_name: str = "performance"
    ) -> Optional[Dict[str, Any]]:
        """Get performance trend for an agent"""
        with self._lock:
            key = f"{agent_name}_{metric_name}"
            values = self.performance_correlations.get(key, [])
            
            if len(values) < 2:
                return None
            
            # Calculate trend
            recent_values = values[-10:] if len(values) >= 10 else values
            older_values = values[-20:-10] if len(values) >= 20 else values[:len(values)//2]
            
            recent_avg = sum(recent_values) / len(recent_values)
            older_avg = sum(older_values) / len(older_values) if older_values else recent_avg
            
            trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
            
            return {
                'agent': agent_name,
                'metric': metric_name,
                'trend': trend,
                'recent_average': recent_avg,
                'change_pct': ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0,
                'data_points': len(values)
            }
    
    def clear_old_activities(self, days: int = 7):
        """Clear activities older than specified days"""
        with self._lock:
            cutoff = datetime.now() - timedelta(days=days)
            original_count = len(self.activities)
            self.activities = [r for r in self.activities if r.timestamp >= cutoff]
            removed = original_count - len(self.activities)
            
            if removed > 0:
                logger.info(f"🧹 Cleared {removed} old activities (older than {days} days)")
                self._save_state()
    
    def _save_state(self):
        """Save tracker state to disk"""
        try:
            data = {
                'activities': [r.to_dict() for r in self.activities[-1000:]],  # Save last 1000
                'agent_stats': {
                    agent: {
                        'total_actions': stats['total_actions'],
                        'action_counts': dict(stats['action_counts']),
                        'success_count': stats['success_count'],
                        'failure_count': stats['failure_count'],
                        'last_activity': stats['last_activity'].isoformat() if stats['last_activity'] else None
                    }
                    for agent, stats in self.agent_stats.items()
                },
                'performance_correlations': {
                    k: v[-100:] for k, v in self.performance_correlations.items()  # Save last 100 per metric
                },
                'agent_memories': self.agent_memories,
                'agent_thinking': self.agent_thinking[-500:],  # Save last 500 thoughts
                'agent_suggestions': self.agent_suggestions[-500:],  # Save last 500 suggestions
                'strategy_info': self.strategy_info[-100:]  # Save last 100 strategy records
            }
            
            filepath = self.storage_dir / "tracker_state.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("💾 Tracker state saved")
        except Exception as e:
            logger.error(f"❌ Failed to save tracker state: {e}")
    
    def _load_state(self):
        """Load tracker state from disk"""
        try:
            filepath = self.storage_dir / "tracker_state.json"
            if not filepath.exists():
                return
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load activities
            self.activities = [
                ActivityRecord.from_dict(r)
                for r in data.get('activities', [])
            ]
            
            # Load agent stats
            for agent, stats in data.get('agent_stats', {}).items():
                self.agent_stats[agent] = {
                    'total_actions': stats['total_actions'],
                    'action_counts': defaultdict(int, stats['action_counts']),
                    'success_count': stats['success_count'],
                    'failure_count': stats['failure_count'],
                    'last_activity': datetime.fromisoformat(stats['last_activity']) if stats['last_activity'] else None
                }
            
            # Load performance correlations
            self.performance_correlations = defaultdict(
                list,
                data.get('performance_correlations', {})
            )
            
            # Load agent memories
            self.agent_memories = data.get('agent_memories', self.agent_memories)
            
            # Load thinking, suggestions, strategy info
            self.agent_thinking = data.get('agent_thinking', [])
            self.agent_suggestions = data.get('agent_suggestions', [])
            self.strategy_info = data.get('strategy_info', [])
            
            logger.info(f"📂 Loaded {len(self.activities)} activities from storage")
        except Exception as e:
            logger.warning(f"⚠️ Could not load tracker state: {e}")
    
    def export_summary(self) -> Dict[str, Any]:
        """Export comprehensive summary of all tracked data"""
        with self._lock:
            # Get basic counts
            total_activities = len(self.activities)
            total_thoughts = len(self.agent_thinking)
            total_suggestions = len(self.agent_suggestions)
            total_strategies = len(self.strategy_info)
            
            # Get agent summaries inline to avoid nested locking
            agents = {}
            for agent, stats in self.agent_stats.items():
                total = stats['total_actions']
                if total == 0:
                    agents[agent] = {'agent': agent, 'no_activity': True}
                else:
                    success_rate = stats['success_count'] / total if total > 0 else 0.0
                    agents[agent] = {
                        'agent': agent,
                        'total_actions': total,
                        'action_breakdown': dict(stats['action_counts']),
                        'success_rate': success_rate,
                        'success_count': stats['success_count'],
                        'failure_count': stats['failure_count'],
                        'last_activity': stats['last_activity'].isoformat() if stats['last_activity'] else None
                    }
        
        # Get recommendations outside lock
        recommendations = self.get_adaptive_recommendations()
        
        return {
            'total_activities': total_activities,
            'total_thoughts': total_thoughts,
            'total_suggestions': total_suggestions,
            'total_strategies': total_strategies,
            'agents': agents,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance
_tracker_instance: Optional[AgentActivityTracker] = None
_tracker_lock = threading.Lock()


def get_tracker_instance(storage_dir: str = "monolith_data") -> AgentActivityTracker:
    """
    Get or create the global tracker instance
    
    Args:
        storage_dir: Directory for storing tracker data
    
    Returns:
        AgentActivityTracker instance
    """
    global _tracker_instance
    
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = AgentActivityTracker(storage_dir)
        return _tracker_instance
