"""
Enhanced Monitoring API Endpoints

This file provides additional Flask endpoints to expose:
1. Trial trades (signals before champion selection)
2. Detailed agent thoughts with context
3. Strategy breakdowns with alpha sources  
4. Individual agent status cards
5. Iteration summaries with results

Usage: Import and register these endpoints with your Flask app
"""

from flask import Flask, jsonify, request
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging


logger = logging.getLogger(__name__)


def register_monolith_endpoints(app: Flask, system, activity_tracker):
    """
    Register enhanced monitoring endpoints for frontend integration
    
    Args:
        app: Flask application instance
        system: E22MinTradesEnforcedSystem instance
        activity_tracker: AgentActivityTracker instance from monolith
    """
    
    @app.route('/api/monolith/trial_trades')
    def get_trial_trades():
        """
        Get all trial trades (signals being tested before champion selection)
        
        Query parameters:
        - iteration: Filter by iteration number
        - limit: Max number of results (default 100)
        - status: Filter by status (viable, rejected, pending)
        """
        iteration = request.args.get('iteration', type=int)
        limit = request.args.get('limit', 100, type=int)
        status_filter = request.args.get('status')
        
        trial_trades = []
        
        # Get strategy info from activity tracker
        strategy_records = activity_tracker.strategy_info
        
        for record in strategy_records[-limit:]:
            if iteration and record.get('iteration') != iteration:
                continue
            
            data = record.get('data', {})
            record_status = data.get('status', 'pending')
            
            if status_filter and record_status != status_filter:
                continue
            
            trial_trades.append({
                'timestamp': record.get('timestamp'),
                'iteration': record.get('iteration'),
                'status': record_status,
                'alpha_sources': data.get('alpha_sources_used', []),
                'signals': data.get('signals', []),
                'execution_optimizations': data.get('execution_optimizations', []),
                'risk_parameters': data.get('risk_parameters', {}),
                'performance': data.get('performance_metrics', {}),
                'rejection_reason': data.get('rejection_reason')
            })
        
        return jsonify({
            'trial_trades': list(reversed(trial_trades)),  # Most recent first
            'total': len(trial_trades),
            'filtered_by': {
                'iteration': iteration,
                'status': status_filter
            }
        })
    
    @app.route('/api/monolith/agent_thoughts')
    def get_agent_thoughts():
        """
        Get detailed agent thoughts with context
        
        Query parameters:
        - agent: Filter by agent name
        - limit: Max number of results (default 50)
        - since_hours: Only show thoughts from last N hours
        """
        agent_filter = request.args.get('agent')
        limit = request.args.get('limit', 50, type=int)
        since_hours = request.args.get('since_hours', type=int)
        
        thoughts = activity_tracker.agent_thinking
        
        # Filter by time if specified
        if since_hours:
            cutoff = datetime.now() - timedelta(hours=since_hours)
            thoughts = [
                t for t in thoughts
                if datetime.fromisoformat(t['timestamp']) >= cutoff
            ]
        
        # Filter by agent if specified
        if agent_filter:
            thoughts = [t for t in thoughts if t['agent'] == agent_filter]
        
        # Limit results
        thoughts = thoughts[-limit:]
        
        return jsonify({
            'thoughts': list(reversed(thoughts)),  # Most recent first
            'total': len(thoughts),
            'filtered_by': {
                'agent': agent_filter,
                'since_hours': since_hours
            }
        })
    
    @app.route('/api/monolith/agent_suggestions')
    def get_agent_suggestions():
        """
        Get agent suggestions for inter-agent communication
        
        Query parameters:
        - type: Filter by suggestion type
        - priority: Filter by priority (low, medium, high)
        - limit: Max number of results (default 50)
        """
        type_filter = request.args.get('type')
        priority_filter = request.args.get('priority')
        limit = request.args.get('limit', 50, type=int)
        
        suggestions = activity_tracker.agent_suggestions
        
        # Filter by type if specified
        if type_filter:
            suggestions = [s for s in suggestions if s['type'] == type_filter]
        
        # Filter by priority if specified
        if priority_filter:
            suggestions = [s for s in suggestions if s['priority'] == priority_filter]
        
        # Limit results
        suggestions = suggestions[-limit:]
        
        return jsonify({
            'suggestions': list(reversed(suggestions)),  # Most recent first
            'total': len(suggestions),
            'filtered_by': {
                'type': type_filter,
                'priority': priority_filter
            }
        })
    
    @app.route('/api/monolith/agent_status')
    def get_agent_status():
        """
        Get individual agent status cards with detailed metrics
        
        Query parameters:
        - agent: Specific agent name (optional, returns all if not specified)
        """
        agent_name = request.args.get('agent')
        
        if agent_name:
            # Return single agent status
            summary = activity_tracker.get_agent_summary(agent_name)
            patterns = activity_tracker.get_activity_patterns(agent_name, time_window_hours=24)
            memory = activity_tracker.get_agent_memory(agent_name)
            
            return jsonify({
                'agent': agent_name,
                'summary': summary,
                'patterns': patterns,
                'memory': memory,
                'recent_activities': [
                    {
                        'timestamp': r.timestamp.isoformat(),
                        'action': r.action,
                        'details': r.details
                    }
                    for r in activity_tracker.get_recent_activities(agent_name, limit=20)
                ]
            })
        else:
            # Return all agent statuses
            all_agents = {}
            for agent in activity_tracker.agent_stats.keys():
                summary = activity_tracker.get_agent_summary(agent)
                if not summary.get('no_activity'):
                    all_agents[agent] = {
                        'summary': summary,
                        'last_activity': summary.get('last_activity'),
                        'success_rate': summary.get('success_rate', 0)
                    }
            
            return jsonify({
                'agents': all_agents,
                'total_agents': len(all_agents),
                'recommendations': activity_tracker.get_adaptive_recommendations()
            })
    
    @app.route('/api/monolith/strategy_breakdown')
    def get_strategy_breakdown():
        """
        Get detailed strategy breakdowns with alpha sources
        
        Query parameters:
        - iteration: Filter by iteration number
        - limit: Max number of results (default 20)
        """
        iteration = request.args.get('iteration', type=int)
        limit = request.args.get('limit', 20, type=int)
        
        strategies = activity_tracker.strategy_info
        
        # Filter by iteration if specified
        if iteration:
            strategies = [s for s in strategies if s.get('iteration') == iteration]
        
        # Limit results
        strategies = strategies[-limit:]
        
        # Enrich with additional context
        enriched = []
        for strat in strategies:
            data = strat.get('data', {})
            enriched.append({
                'timestamp': strat.get('timestamp'),
                'iteration': strat.get('iteration'),
                'alpha_sources_used': data.get('alpha_sources_used', []),
                'market_regime': data.get('market_regime'),
                'execution_optimizations': data.get('execution_optimizations', []),
                'risk_parameters': data.get('risk_parameters', {}),
                'performance_metrics': data.get('performance_metrics', {}),
                'trade_count': data.get('trade_count', 0),
                'status': data.get('status', 'unknown'),
                'signals_generated': len(data.get('signals', []))
            })
        
        return jsonify({
            'strategies': list(reversed(enriched)),  # Most recent first
            'total': len(enriched),
            'filtered_by': {
                'iteration': iteration
            }
        })
    
    @app.route('/api/monolith/iteration_summary')
    def get_iteration_summary():
        """
        Get comprehensive iteration summaries with all results
        
        Query parameters:
        - iteration: Specific iteration number (optional)
        - limit: Max number of results if not specific (default 10)
        """
        iteration = request.args.get('iteration', type=int)
        limit = request.args.get('limit', 10, type=int)
        
        if iteration:
            # Return specific iteration details
            iter_data = system.monitoring_agent.iteration_data.get(iteration, {})
            
            # Get all related data for this iteration
            thoughts = [
                t for t in activity_tracker.agent_thinking
                if t.get('context', {}).get('iteration') == iteration
            ]
            
            strategies = [
                s for s in activity_tracker.strategy_info
                if s.get('iteration') == iteration
            ]
            
            suggestions = [
                s for s in activity_tracker.agent_suggestions
                if s.get('details', {}).get('iteration') == iteration or
                   (datetime.fromisoformat(s['timestamp']) >= 
                    datetime.fromisoformat(iter_data.get('start_time', datetime.now().isoformat())) if iter_data.get('start_time') else False)
            ]
            
            return jsonify({
                'iteration': iteration,
                'start_time': iter_data.get('start_time'),
                'end_time': iter_data.get('end_time'),
                'completed': iter_data.get('completed', False),
                'results': iter_data.get('results', {}),
                'phases': iter_data.get('phases', []),
                'agents_created': iter_data.get('agents_created', []),
                'performance_metrics': iter_data.get('performance_metrics', {}),
                'agent_thoughts': thoughts[-50:],  # Last 50 thoughts
                'strategies_tested': len(strategies),
                'strategy_details': strategies,
                'agent_suggestions': suggestions[-30:],  # Last 30 suggestions
                'trial_trades': [s.get('data', {}).get('signals', []) for s in strategies]
            })
        else:
            # Return summary of recent iterations
            recent_iterations = []
            iter_nums = sorted(system.monitoring_agent.iteration_data.keys(), reverse=True)[:limit]
            
            for iter_num in iter_nums:
                iter_data = system.monitoring_agent.iteration_data[iter_num]
                strategies = [
                    s for s in activity_tracker.strategy_info
                    if s.get('iteration') == iter_num
                ]
                
                recent_iterations.append({
                    'iteration': iter_num,
                    'start_time': iter_data.get('start_time'),
                    'end_time': iter_data.get('end_time'),
                    'completed': iter_data.get('completed', False),
                    'strategies_tested': len(strategies),
                    'performance': iter_data.get('performance_metrics', {}),
                    'status': iter_data.get('results', {}).get('status', 'unknown')
                })
            
            return jsonify({
                'iterations': recent_iterations,
                'total': len(system.monitoring_agent.iteration_data)
            })
    
    @app.route('/api/monolith/dashboard_data')
    def get_dashboard_data():
        """
        Get comprehensive dashboard data for frontend
        Combines all monitoring data into one endpoint for efficiency
        """
        # Get recent iteration
        current_iter = system.iteration_count
        
        # Get agent status for all agents
        agent_statuses = {}
        for agent in activity_tracker.agent_stats.keys():
            summary = activity_tracker.get_agent_summary(agent)
            if not summary.get('no_activity'):
                agent_statuses[agent] = summary
        
        # Get recent thoughts
        recent_thoughts = activity_tracker.agent_thinking[-20:]
        
        # Get recent suggestions
        recent_suggestions = activity_tracker.agent_suggestions[-20:]
        
        # Get recent strategies
        recent_strategies = activity_tracker.strategy_info[-10:]
        
        # Get trial trades from recent strategies
        trial_trades = []
        for strat in recent_strategies:
            data = strat.get('data', {})
            signals = data.get('signals', [])
            if signals:
                trial_trades.extend([
                    {
                        'timestamp': strat.get('timestamp'),
                        'iteration': strat.get('iteration'),
                        'signal': sig,
                        'status': data.get('status', 'pending')
                    }
                    for sig in signals[:10]  # First 10 signals per strategy
                ])
        
        # Get recommendations
        recommendations = activity_tracker.get_adaptive_recommendations()
        
        return jsonify({
            'current_iteration': current_iter,
            'system_status': {
                'champions': len(system.active_champions),
                'errors': len(system.error_memory),
                'uptime_seconds': int(time.time() - system.heartbeat_monitor.stats.get('uptime_start', time.time()))
            },
            'agents': agent_statuses,
            'recent_thoughts': list(reversed(recent_thoughts)),
            'recent_suggestions': list(reversed(recent_suggestions)),
            'recent_strategies': list(reversed(recent_strategies)),
            'trial_trades': list(reversed(trial_trades)),
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    
    logger.info("✅ Registered Monolith enhanced API endpoints")
    logger.info("   - /api/monolith/trial_trades")
    logger.info("   - /api/monolith/agent_thoughts")
    logger.info("   - /api/monolith/agent_suggestions")
    logger.info("   - /api/monolith/agent_status")
    logger.info("   - /api/monolith/strategy_breakdown")
    logger.info("   - /api/monolith/iteration_summary")
    logger.info("   - /api/monolith/dashboard_data")


def get_endpoint_documentation() -> Dict[str, Any]:
    """
    Return documentation for all Monolith API endpoints
    """
    return {
        'endpoints': [
            {
                'path': '/api/monolith/trial_trades',
                'method': 'GET',
                'description': 'Get all trial trades (signals being tested before champion selection)',
                'query_params': {
                    'iteration': 'Filter by iteration number (int)',
                    'limit': 'Max number of results (default 100)',
                    'status': 'Filter by status (viable, rejected, pending)'
                },
                'response': 'List of trial trade records with signals, alpha sources, and performance'
            },
            {
                'path': '/api/monolith/agent_thoughts',
                'method': 'GET',
                'description': 'Get detailed agent thoughts with context',
                'query_params': {
                    'agent': 'Filter by agent name (str)',
                    'limit': 'Max number of results (default 50)',
                    'since_hours': 'Only show thoughts from last N hours (int)'
                },
                'response': 'List of agent thinking records with context'
            },
            {
                'path': '/api/monolith/agent_suggestions',
                'method': 'GET',
                'description': 'Get agent suggestions for inter-agent communication',
                'query_params': {
                    'type': 'Filter by suggestion type (str)',
                    'priority': 'Filter by priority (low, medium, high)',
                    'limit': 'Max number of results (default 50)'
                },
                'response': 'List of agent suggestions'
            },
            {
                'path': '/api/monolith/agent_status',
                'method': 'GET',
                'description': 'Get individual agent status cards with detailed metrics',
                'query_params': {
                    'agent': 'Specific agent name (optional, returns all if not specified)'
                },
                'response': 'Agent status with summary, patterns, memory, and recent activities'
            },
            {
                'path': '/api/monolith/strategy_breakdown',
                'method': 'GET',
                'description': 'Get detailed strategy breakdowns with alpha sources',
                'query_params': {
                    'iteration': 'Filter by iteration number (int)',
                    'limit': 'Max number of results (default 20)'
                },
                'response': 'List of strategy records with alpha sources and performance metrics'
            },
            {
                'path': '/api/monolith/iteration_summary',
                'method': 'GET',
                'description': 'Get comprehensive iteration summaries with all results',
                'query_params': {
                    'iteration': 'Specific iteration number (int, optional)',
                    'limit': 'Max number of results if not specific (default 10)'
                },
                'response': 'Iteration details including thoughts, strategies, and trial trades'
            },
            {
                'path': '/api/monolith/dashboard_data',
                'method': 'GET',
                'description': 'Get comprehensive dashboard data for frontend (efficient single call)',
                'query_params': {},
                'response': 'Combined data: agents, thoughts, suggestions, strategies, trial trades, recommendations'
            }
        ],
        'integration_example': '''
# In your Flask app initialization:
from monolith import get_tracker_instance
from monolith_api_endpoints import register_monolith_endpoints

# Get tracker instance
tracker = get_tracker_instance()

# Register enhanced endpoints
register_monolith_endpoints(app, system, tracker)

# Frontend can now call:
# - GET /api/monolith/trial_trades?iteration=5&status=viable
# - GET /api/monolith/agent_thoughts?agent=ResearchQuant&limit=100
# - GET /api/monolith/dashboard_data (for full dashboard in one call)
'''
    }
