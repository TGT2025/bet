"""
Agent Activity Tracker - Exposes all agent thoughts, suggestions, and activities to frontend
Creates real-time feed of what each agent is doing and thinking
"""

import os
import json
import time
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict


class AgentActivityTracker:
    """Tracks and exposes all agent activities for frontend visualization"""
    
    def __init__(self, activity_log_file: str = "agent_activity.json"):
        self.activity_log_file = activity_log_file
        self.current_activities = []
        self.agent_thoughts = defaultdict(list)
        self.strategy_details = {}
        self.iteration_summary = {}
        
    def log_agent_thinking(self, agent_name: str, thought: str, context: Dict[str, Any] = None):
        """
        Log what an agent is thinking
        
        Args:
            agent_name: Name of agent (ResearchQuant, ExecutionQuant, RiskQuant, Reasoner, Coder, etc.)
            thought: What the agent is thinking/doing
            context: Additional context data
        """
        activity = {
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "agent": agent_name,
            "thought": thought,
            "context": context or {},
            "type": "thinking"
        }
        
        self.current_activities.append(activity)
        self.agent_thoughts[agent_name].append(activity)
        self._save_to_file()
        
    def log_agent_suggestion(self, agent_name: str, suggestion_type: str, suggestion: Dict[str, Any]):
        """
        Log agent suggestions (alpha sources, optimizations, risk limits, etc.)
        
        Args:
            agent_name: Name of agent
            suggestion_type: Type of suggestion (alpha_source, optimization, risk_limit, etc.)
            suggestion: The actual suggestion data
        """
        activity = {
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "agent": agent_name,
            "suggestion_type": suggestion_type,
            "suggestion": suggestion,
            "type": "suggestion"
        }
        
        self.current_activities.append(activity)
        self.agent_thoughts[agent_name].append(activity)
        self._save_to_file()
        
    def log_strategy_info(self, iteration: int, strategy_data: Dict[str, Any]):
        """
        Log detailed strategy information
        
        Args:
            iteration: Iteration number
            strategy_data: Complete strategy details including:
                - alpha_sources_used
                - execution_optimizations_applied
                - risk_parameters
                - expected_performance
                - code_files_generated
        """
        strategy_info = {
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "iteration": iteration,
            "strategy_data": strategy_data,
            "type": "strategy_info"
        }
        
        self.strategy_details[iteration] = strategy_info
        self.current_activities.append(strategy_info)
        self._save_to_file()
        
    def log_iteration_summary(self, iteration: int, summary: Dict[str, Any]):
        """
        Log iteration summary with all agent contributions
        
        Args:
            iteration: Iteration number
            summary: Summary including:
                - research_agent_inputs
                - execution_agent_optimizations
                - risk_agent_validations
                - reasoner_decisions
                - coder_outputs
                - final_results
        """
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "unix_time": time.time(),
            "iteration": iteration,
            "summary": summary,
            "type": "iteration_summary"
        }
        
        self.iteration_summary[iteration] = summary_data
        self.current_activities.append(summary_data)
        self._save_to_file()
        
    def get_recent_activities(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent agent activities"""
        return self.current_activities[-limit:]
    
    def get_agent_thoughts(self, agent_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent thoughts from a specific agent"""
        return self.agent_thoughts[agent_name][-limit:]
    
    def get_all_agent_status(self) -> Dict[str, Any]:
        """Get current status of all agents"""
        status = {}
        for agent_name, thoughts in self.agent_thoughts.items():
            if thoughts:
                latest = thoughts[-1]
                status[agent_name] = {
                    "last_active": latest["timestamp"],
                    "last_thought": latest.get("thought", latest.get("suggestion_type", "Unknown")),
                    "activity_count": len(thoughts)
                }
        return status
    
    def get_strategy_details(self, iteration: int = None) -> Dict[str, Any]:
        """Get strategy details for an iteration (or latest)"""
        if iteration is None:
            # Get latest
            if self.strategy_details:
                latest_iter = max(self.strategy_details.keys())
                return self.strategy_details[latest_iter]
            return {}
        return self.strategy_details.get(iteration, {})
    
    def get_iteration_summary(self, iteration: int = None) -> Dict[str, Any]:
        """Get iteration summary (or latest)"""
        if iteration is None:
            if self.iteration_summary:
                latest_iter = max(self.iteration_summary.keys())
                return self.iteration_summary[latest_iter]
            return {}
        return self.iteration_summary.get(iteration, {})
    
    def get_frontend_dashboard_data(self) -> Dict[str, Any]:
        """
        Get all data needed for frontend dashboard in one call
        
        Returns comprehensive view of:
        - Current agent status
        - Recent activities
        - Latest strategy details
        - Latest iteration summary
        - Agent suggestions breakdown
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "agent_status": self.get_all_agent_status(),
            "recent_activities": self.get_recent_activities(50),
            "latest_strategy": self.get_strategy_details(),
            "latest_iteration": self.get_iteration_summary(),
            "agent_thoughts_summary": {
                agent: thoughts[-5:] if len(thoughts) > 5 else thoughts
                for agent, thoughts in self.agent_thoughts.items()
            },
            "total_iterations": len(self.iteration_summary),
            "total_activities": len(self.current_activities)
        }
    
    def _save_to_file(self):
        """Save current activities to file for persistence"""
        try:
            data = {
                "activities": self.current_activities[-500:],  # Keep last 500
                "agent_thoughts": {
                    agent: thoughts[-100:] for agent, thoughts in self.agent_thoughts.items()
                },
                "strategy_details": self.strategy_details,
                "iteration_summary": self.iteration_summary,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.activity_log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save agent activity log: {e}")
    
    def load_from_file(self):
        """Load previous activities from file"""
        if os.path.exists(self.activity_log_file):
            try:
                with open(self.activity_log_file, 'r') as f:
                    data = json.load(f)
                
                self.current_activities = data.get("activities", [])
                self.agent_thoughts = defaultdict(list, data.get("agent_thoughts", {}))
                self.strategy_details = {
                    int(k): v for k, v in data.get("strategy_details", {}).items()
                }
                self.iteration_summary = {
                    int(k): v for k, v in data.get("iteration_summary", {}).items()
                }
                
                print(f"Loaded {len(self.current_activities)} previous activities")
            except Exception as e:
                print(f"Warning: Failed to load agent activity log: {e}")
