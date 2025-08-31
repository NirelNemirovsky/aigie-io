"""
LangGraph interceptor for real-time error detection and monitoring.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Union, List
from datetime import datetime

from ..core.error_detector import ErrorDetector
from ..core.error_types import ErrorContext
from ..reporting.logger import AigieLogger


class LangGraphInterceptor:
    """Intercepts LangGraph operations to detect errors and monitor performance."""
    
    def __init__(self, error_detector: ErrorDetector, logger: AigieLogger):
        self.error_detector = error_detector
        self.logger = logger
        self.intercepted_classes = set()
        self.original_methods = {}
        
        # LangGraph components to intercept
        self.target_classes = {
            'StateGraph': ['add_node', 'add_edge', 'compile', 'set_entry_point', 'set_finish_point'],
            'Graph': ['add_node', 'add_edge', 'compile', 'set_entry_point', 'set_finish_point'],
            'Node': ['__call__', 'invoke', 'ainvoke'],
            'StateNode': ['__call__', 'invoke', 'ainvoke'],
            'Checkpointer': ['get', 'put', 'list_keys', 'delete'],
            'MemorySaver': ['save', 'load', 'clear'],
            'StateGraphApp': ['invoke', 'ainvoke', 'stream', 'astream'],
            'CompiledGraph': ['invoke', 'ainvoke', 'stream', 'astream'],
        }
        
        # Track graph state and transitions
        self.graph_states = {}
        self.node_executions = {}
        self.state_transitions = []
    
    def start_intercepting(self):
        """Start intercepting LangGraph operations."""
        self.error_detector.start_monitoring()
        self.logger.log_system_event("Started LangGraph interception")
        
        # Intercept existing instances
        self._intercept_existing_instances()
        
        # Patch class methods for future instances
        self._patch_classes()
    
    def stop_intercepting(self):
        """Stop intercepting LangGraph operations."""
        self.error_detector.stop_monitoring()
        self.logger.log_system_event("Stopped LangGraph interception")
        
        # Restore original methods
        self._restore_original_methods()
    
    def _intercept_existing_instances(self):
        """Intercept existing LangGraph instances."""
        # This would require access to a registry of instances
        # For now, we'll focus on patching classes for future instances
        pass
    
    def _patch_classes(self):
        """Patch LangGraph classes to intercept method calls."""
        try:
            # Import LangGraph classes dynamically
            self._patch_langgraph_classes()
        except ImportError as e:
            self.logger.log_system_event(f"Could not import LangGraph classes: {e}")
    
    def _patch_langgraph_classes(self):
        """Patch specific LangGraph classes."""
        # Try to import and patch LangGraph classes
        try:
            from langgraph.graph import StateGraph, Graph
            from langgraph.graph.message import add_messages
            from langgraph.checkpoint import BaseCheckpointer
            from langgraph.checkpoint.memory import MemorySaver
            from langgraph.graph import CompiledGraph
            
            classes_to_patch = {
                'StateGraph': StateGraph,
                'Graph': Graph,
                'BaseCheckpointer': BaseCheckpointer,
                'MemorySaver': MemorySaver,
                'CompiledGraph': CompiledGraph,
            }
            
            for class_name, cls in classes_to_patch.items():
                if cls and class_name in self.target_classes:
                    self._patch_class_methods(cls, class_name)
                    
        except ImportError as e:
            self.logger.log_system_event(f"LangGraph not available: {e}")
    
    def _patch_class_methods(self, cls: type, class_name: str):
        """Patch methods of a specific class."""
        if class_name in self.intercepted_classes:
            return
        
        methods_to_patch = self.target_classes.get(class_name, [])
        
        for method_name in methods_to_patch:
            if hasattr(cls, method_name):
                original_method = getattr(cls, method_name)
                
                # Store original method
                key = f"{cls.__name__}.{method_name}"
                self.original_methods[key] = original_method
                
                # Create patched method
                if inspect.iscoroutinefunction(original_method):
                    patched_method = self._create_async_patched_method(original_method, class_name, method_name)
                else:
                    patched_method = self._create_sync_patched_method(original_method, class_name, method_name)
                
                # Apply the patch
                setattr(cls, method_name, patched_method)
        
        self.intercepted_classes.add(class_name)
        self.logger.log_system_event(f"Patched {class_name} methods: {methods_to_patch}")
    
    def _create_sync_patched_method(self, original_method: Callable, class_name: str, method_name: str):
        """Create a synchronous patched method."""
        @functools.wraps(original_method)
        def patched_method(self_instance, *args, **kwargs):
            # Create error context
            context = ErrorContext(
                timestamp=datetime.now(),
                framework="langgraph",
                component=class_name,
                method=method_name,
                input_data=self._extract_input_data(args, kwargs, method_name),
                state=self._extract_state_data(self_instance, method_name)
            )
            
            # Monitor execution
            with self.error_detector.monitor_execution(
                framework="langgraph",
                component=class_name,
                method=method_name,
                input_data=context.input_data,
                state=context.state
            ):
                try:
                    # Call original method
                    result = original_method(self_instance, *args, **kwargs)
                    
                    # Track state changes for StateGraph operations
                    if class_name == 'StateGraph' and method_name in ['add_node', 'add_edge', 'compile']:
                        self._track_graph_changes(self_instance, method_name, args, kwargs, result)
                    
                    return result
                except Exception as e:
                    # Error will be detected by the context manager
                    raise
        
        return patched_method
    
    def _create_async_patched_method(self, original_method: Callable, class_name: str, method_name: str):
        """Create an asynchronous patched method."""
        @functools.wraps(original_method)
        async def patched_method(self_instance, *args, **kwargs):
            # Create error context
            context = ErrorContext(
                timestamp=datetime.now(),
                framework="langgraph",
                component=class_name,
                method=method_name,
                input_data=self._extract_input_data(args, kwargs, method_name),
                state=self._extract_state_data(self_instance, method_name)
            )
            
            # Monitor execution
            async with self.error_detector.monitor_execution_async(
                framework="langgraph",
                component=class_name,
                method=method_name,
                input_data=context.input_data,
                state=context.state
            ):
                try:
                    # Call original method
                    result = await original_method(self_instance, *args, **kwargs)
                    
                    # Track state changes for StateGraph operations
                    if class_name == 'StateGraph' and method_name in ['add_node', 'add_edge', 'compile']:
                        self._track_graph_changes(self_instance, method_name, args, kwargs, result)
                    
                    return result
                except Exception as e:
                    # Error will be detected by the context manager
                    raise
        
        return patched_method
    
    def _extract_input_data(self, args: tuple, kwargs: dict, method_name: str) -> Optional[Dict[str, Any]]:
        """Extract relevant input data for monitoring."""
        input_data = {}
        
        # Extract common input parameters
        if args:
            if method_name in ['add_node', 'add_edge']:
                input_data['node_name'] = str(args[0]) if args else "unknown"
            elif method_name in ['invoke', 'ainvoke']:
                input_data['input_data'] = str(args[0])[:200] if args else "unknown"
        
        # Extract keyword arguments
        for key, value in kwargs.items():
            if key in ['input', 'inputs', 'state', 'config']:
                input_data[key] = str(value)[:200]  # Truncate long inputs
            elif key in ['checkpointer', 'memory', 'callbacks']:
                input_data[key] = type(value).__name__  # Just the type
        
        return input_data if input_data else None
    
    def _extract_state_data(self, instance: Any, method_name: str) -> Optional[Dict[str, Any]]:
        """Extract state data from LangGraph instances."""
        state_data = {}
        
        try:
            if hasattr(instance, 'nodes'):
                state_data['node_count'] = len(instance.nodes)
            
            if hasattr(instance, 'edges'):
                state_data['edge_count'] = len(instance.edges)
            
            if hasattr(instance, 'entry_point'):
                state_data['entry_point'] = instance.entry_point
            
            if hasattr(instance, 'finish_point'):
                state_data['finish_point'] = instance.finish_point
                
        except Exception:
            # Ignore errors in state extraction
            pass
        
        return state_data if state_data else None
    
    def _track_graph_changes(self, instance: Any, method_name: str, args: tuple, kwargs: dict, result: Any):
        """Track changes to graph structure."""
        graph_id = id(instance)
        
        if graph_id not in self.graph_states:
            self.graph_states[graph_id] = {
                'nodes': set(),
                'edges': set(),
                'entry_point': None,
                'finish_point': None,
                'last_modified': datetime.now()
            }
        
        state = self.graph_states[graph_id]
        state['last_modified'] = datetime.now()
        
        if method_name == 'add_node':
            node_name = args[0] if args else "unknown"
            state['nodes'].add(node_name)
            self.logger.log_system_event(f"Added node: {node_name}", {"graph_id": graph_id})
            
        elif method_name == 'add_edge':
            if len(args) >= 2:
                from_node = args[0]
                to_node = args[1]
                edge = (from_node, to_node)
                state['edges'].add(edge)
                self.logger.log_system_event(f"Added edge: {from_node} -> {to_node}", {"graph_id": graph_id})
        
        elif method_name == 'compile':
            self.logger.log_system_event("Graph compiled", {
                "graph_id": graph_id,
                "node_count": len(state['nodes']),
                "edge_count": len(state['edges'])
            })
    
    def intercept_node_execution(self, node_name: str, node_func: Callable):
        """Intercept a specific node function."""
        @functools.wraps(node_func)
        def intercepted_node(*args, **kwargs):
            # Create error context
            context = ErrorContext(
                timestamp=datetime.now(),
                framework="langgraph",
                component="Node",
                method="__call__",
                input_data=self._extract_node_input(args, kwargs),
                state={"node_name": node_name}
            )
            
            # Monitor execution
            with self.error_detector.monitor_execution(
                framework="langgraph",
                component="Node",
                method="__call__",
                input_data=context.input_data,
                state=context.state
            ):
                try:
                    # Track node execution
                    self._track_node_execution(node_name, "start")
                    
                    # Call original function
                    result = node_func(*args, **kwargs)
                    
                    # Track successful completion
                    self._track_node_execution(node_name, "complete")
                    
                    return result
                except Exception as e:
                    # Track error
                    self._track_node_execution(node_name, "error", str(e))
                    # Error will be detected by the context manager
                    raise
        
        return intercepted_node
    
    def _extract_node_input(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract input data for node execution."""
        input_data = {}
        
        if args:
            # First argument is usually the state
            if args[0] and hasattr(args[0], '__dict__'):
                state_keys = list(args[0].__dict__.keys())[:5]  # Limit to first 5 keys
                input_data['state_keys'] = state_keys
                input_data['state_type'] = type(args[0]).__name__
        
        # Extract keyword arguments
        for key, value in kwargs.items():
            if key in ['config', 'callbacks']:
                input_data[key] = type(value).__name__
        
        return input_data
    
    def _track_node_execution(self, node_name: str, status: str, error_message: str = None):
        """Track node execution status."""
        timestamp = datetime.now()
        
        if node_name not in self.node_executions:
            self.node_executions[node_name] = []
        
        execution_record = {
            'timestamp': timestamp,
            'status': status,
            'error_message': error_message
        }
        
        self.node_executions[node_name].append(execution_record)
        
        # Keep only last 100 executions per node
        if len(self.node_executions[node_name]) > 100:
            self.node_executions[node_name] = self.node_executions[node_name][-100:]
    
    def track_state_transition(self, from_node: str, to_node: str, state_data: Dict[str, Any]):
        """Track state transitions between nodes."""
        transition = {
            'timestamp': datetime.now(),
            'from_node': from_node,
            'to_node': to_node,
            'state_keys': list(state_data.keys()) if state_data else [],
            'state_size': len(str(state_data)) if state_data else 0
        }
        
        self.state_transitions.append(transition)
        
        # Keep only last 1000 transitions
        if len(self.state_transitions) > 1000:
            self.state_transitions = self.state_transitions[-1000:]
        
        self.logger.log_system_event(
            f"State transition: {from_node} -> {to_node}",
            {
                "state_keys": transition['state_keys'],
                "state_size": transition['state_size']
            }
        )
    
    def _restore_original_methods(self):
        """Restore original methods to classes."""
        for key, original_method in self.original_methods.items():
            try:
                class_name, method_name = key.split('.')
                
                # Find the class
                for cls in self.intercepted_classes:
                    if cls.__name__ == class_name:
                        setattr(cls, method_name, original_method)
                        break
                        
            except Exception as e:
                self.logger.log_system_event(f"Failed to restore {key}: {e}")
        
        self.original_methods.clear()
        self.intercepted_classes.clear()
    
    def get_interception_status(self) -> Dict[str, Any]:
        """Get current interception status."""
        return {
            "is_intercepting": self.error_detector.is_monitoring,
            "intercepted_classes": list(self.intercepted_classes),
            "patched_methods": list(self.original_methods.keys()),
            "target_classes": list(self.target_classes.keys()),
            "tracked_graphs": len(self.graph_states),
            "tracked_nodes": len(self.node_executions),
            "state_transitions": len(self.state_transitions)
        }
    
    def get_graph_analysis(self) -> Dict[str, Any]:
        """Get analysis of intercepted graphs."""
        analysis = {
            "total_graphs": len(self.graph_states),
            "graphs": {}
        }
        
        for graph_id, state in self.graph_states.items():
            analysis["graphs"][str(graph_id)] = {
                "node_count": len(state['nodes']),
                "edge_count": len(state['edges']),
                "nodes": list(state['nodes']),
                "edges": [f"{from_node} -> {to_node}" for from_node, to_node in state['edges']],
                "last_modified": state['last_modified'].isoformat()
            }
        
        return analysis
    
    def get_node_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about node executions."""
        stats = {
            "total_nodes": len(self.node_executions),
            "nodes": {}
        }
        
        for node_name, executions in self.node_executions.items():
            total_executions = len(executions)
            successful = len([e for e in executions if e['status'] == 'complete'])
            errors = len([e for e in executions if e['status'] == 'error'])
            
            stats["nodes"][node_name] = {
                "total_executions": total_executions,
                "successful": successful,
                "errors": errors,
                "success_rate": (successful / total_executions * 100) if total_executions > 0 else 0,
                "last_execution": executions[-1]['timestamp'].isoformat() if executions else None
            }
        
        return stats
    
    def get_state_transition_analysis(self) -> Dict[str, Any]:
        """Get analysis of state transitions."""
        if not self.state_transitions:
            return {"total_transitions": 0}
        
        # Analyze transition patterns
        transition_counts = {}
        for transition in self.state_transitions:
            key = f"{transition['from_node']} -> {transition['to_node']}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Find most common transitions
        most_common = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_transitions": len(self.state_transitions),
            "unique_transitions": len(transition_counts),
            "most_common_transitions": most_common,
            "recent_transitions": [
                {
                    "from": t['from_node'],
                    "to": t['to_node'],
                    "timestamp": t['timestamp'].isoformat(),
                    "state_size": t['state_size']
                }
                for t in self.state_transitions[-10:]  # Last 10 transitions
            ]
        }
