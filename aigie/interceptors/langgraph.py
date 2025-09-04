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
        
        # LangGraph components to intercept (updated for modern LangGraph)
        self.target_classes = {
            # Core Graph Components
            'StateGraph': ['add_node', 'add_edge', 'add_conditional_edges', 'compile', 'set_entry_point', 'set_finish_point'],
            'MessageGraph': ['add_node', 'add_edge', 'add_conditional_edges', 'compile', 'set_entry_point', 'set_finish_point'],
            
            # Compiled Graph (main execution interface)
            'CompiledStateGraph': ['invoke', 'ainvoke', 'stream', 'astream', 'stream_events', 'astream_events'],
            'CompiledGraph': ['invoke', 'ainvoke', 'stream', 'astream', 'stream_events', 'astream_events'],
            
            # Checkpointer System
            'MemorySaver': ['put', 'aget', 'aput', 'alist', 'get_tuple', 'put_writes'],
            'SqliteSaver': ['put', 'aget', 'aput', 'alist', 'get_tuple', 'put_writes'],
            'BaseCheckpointSaver': ['put', 'aget', 'aput', 'alist', 'get_tuple', 'put_writes'],
            
            # Prebuilt Agents
            'create_react_agent': ['invoke', 'ainvoke', 'stream', 'astream', 'stream_events', 'astream_events'],
            
            # Human Input Nodes
            'HumanMessage': ['__init__'],
            
            # Thread Management
            'Thread': ['invoke', 'ainvoke', 'stream', 'astream'],
        }
        
        # Track graph state and transitions
        self.graph_states = {}
        self.node_executions = {}
        self.state_transitions = []
        
        # Event streaming monitoring
        self.streaming_sessions = {}
        self.event_history = []
        
        # Checkpoint monitoring
        self.checkpoint_operations = []
        
        # Human-in-the-loop tracking
        self.human_interactions = []
    
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
        classes_to_patch = {}
        
        # Core Graph Components
        try:
            from langgraph.graph import StateGraph, MessageGraph
            classes_to_patch.update({
                'StateGraph': StateGraph,
                'MessageGraph': MessageGraph,
            })
        except ImportError:
            self.logger.log_system_event("Core LangGraph classes not available")
        
        # Compiled Graph Components
        try:
            from langgraph.graph.state import CompiledStateGraph
            classes_to_patch['CompiledStateGraph'] = CompiledStateGraph
            classes_to_patch['CompiledGraph'] = CompiledStateGraph  # Alias for backwards compatibility
        except ImportError:
            self.logger.log_system_event("CompiledStateGraph not available")
        
        # Checkpointer System
        try:
            from langgraph.checkpoint.memory import MemorySaver
            classes_to_patch['MemorySaver'] = MemorySaver
        except ImportError:
            self.logger.log_system_event("MemorySaver not available")
        
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            classes_to_patch['SqliteSaver'] = SqliteSaver
        except ImportError:
            self.logger.log_system_event("SqliteSaver not available")
        
        try:
            from langgraph.checkpoint.base import BaseCheckpointSaver
            classes_to_patch['BaseCheckpointSaver'] = BaseCheckpointSaver
        except ImportError:
            self.logger.log_system_event("BaseCheckpointSaver not available")
        
        # Prebuilt Agents
        try:
            from langgraph.prebuilt import create_react_agent
            # Note: create_react_agent is a function, we'll handle it separately
            self.logger.log_system_event("create_react_agent available for interception")
        except ImportError:
            self.logger.log_system_event("create_react_agent not available")
        
        # Human Input Components
        try:
            from langchain_core.messages import HumanMessage
            classes_to_patch['HumanMessage'] = HumanMessage
        except ImportError:
            self.logger.log_system_event("HumanMessage not available")
        
        # Patch all available classes
        for class_name, cls in classes_to_patch.items():
            if cls and class_name in self.target_classes:
                self._patch_class_methods(cls, class_name)
    
    def _patch_class_methods(self, cls: type, class_name: str):
        """Patch methods of a specific class."""
        if class_name in self.intercepted_classes:
            return
        
        methods_to_patch = self.target_classes.get(class_name, [])
        
        for method_name in methods_to_patch:
            if hasattr(cls, method_name):
                method_descriptor = getattr(cls, method_name)
                
                # Store original method
                key = f"{cls.__name__}.{method_name}"
                self.original_methods[key] = method_descriptor
                
                # Create patched method
                if inspect.iscoroutinefunction(method_descriptor):
                    patched_method = self._create_async_patched_method(method_descriptor, class_name, method_name)
                else:
                    patched_method = self._create_sync_patched_method(method_descriptor, class_name, method_name)
                
                # Apply the patch
                setattr(cls, method_name, patched_method)
        
        self.intercepted_classes.add(cls)
        self.logger.log_system_event(f"Patched {class_name} methods: {methods_to_patch}")
    
    def _create_sync_patched_method(self, original_method: Callable, class_name: str, method_name: str):
        """Create a synchronous patched method."""
        # Only use functools.wraps if original_method is actually callable and has the required attributes
        if callable(original_method) and hasattr(original_method, '__name__'):
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
                
                # Store operation for potential retry
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + args, kwargs, context
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
        else:
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
                
                # Store operation for potential retry
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + args, kwargs, context
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
        # Only use functools.wraps if original_method is actually callable and has the required attributes
        if callable(original_method) and hasattr(original_method, '__name__'):
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
                
                # Store operation for potential retry
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + args, kwargs, context
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
        else:
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
                
                # Store operation for potential retry
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + args, kwargs, context
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
                        # Ensure original_method is actually callable before restoring
                        if callable(original_method):
                            setattr(cls, method_name, original_method)
                        else:
                            self.logger.log_system_event(f"Original method for {key} is not callable: {type(original_method)}")
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
            "state_transitions": len(self.state_transitions),
            "streaming_sessions": len(self.streaming_sessions),
            "active_streams": len([s for s in self.streaming_sessions.values() if s['status'] == 'active']),
            "event_history_size": len(self.event_history),
            "checkpoint_operations": len(self.checkpoint_operations),
            "human_interactions": len(self.human_interactions)
        }
    
    def get_streaming_analysis(self) -> Dict[str, Any]:
        """Get analysis of streaming sessions and events."""
        active_sessions = [s for s in self.streaming_sessions.values() if s['status'] == 'active']
        completed_sessions = [s for s in self.streaming_sessions.values() if s['status'] == 'completed']
        error_sessions = [s for s in self.streaming_sessions.values() if s['status'] == 'error']
        
        # Analyze recent events
        recent_events = self.event_history[-50:] if self.event_history else []
        event_types = {}
        for event in recent_events:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            "total_sessions": len(self.streaming_sessions),
            "active_sessions": len(active_sessions),
            "completed_sessions": len(completed_sessions),
            "error_sessions": len(error_sessions),
            "total_events": len(self.event_history),
            "recent_event_types": event_types,
            "active_session_details": [
                {
                    "start_time": s['start_time'].isoformat(),
                    "method": s['method'],
                    "duration_seconds": (datetime.now() - s['start_time']).total_seconds()
                }
                for s in active_sessions
            ]
        }
    
    def get_checkpoint_analysis(self) -> Dict[str, Any]:
        """Get analysis of checkpoint operations."""
        if not self.checkpoint_operations:
            return {"total_operations": 0}
        
        successful_ops = [op for op in self.checkpoint_operations if op['status'] == 'success']
        failed_ops = [op for op in self.checkpoint_operations if op['status'] == 'error']
        
        # Analyze operation types
        operation_types = {}
        for op in self.checkpoint_operations:
            method = op['method']
            operation_types[method] = operation_types.get(method, 0) + 1
        
        return {
            "total_operations": len(self.checkpoint_operations),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": (len(successful_ops) / len(self.checkpoint_operations) * 100) if self.checkpoint_operations else 0,
            "operation_types": operation_types,
            "recent_operations": [
                {
                    "method": op['method'],
                    "status": op['status'],
                    "checkpointer_type": op['checkpointer_type'],
                    "timestamp": op['start_time'].isoformat()
                }
                for op in self.checkpoint_operations[-10:]  # Last 10 operations
            ]
        }
    
    def get_human_interaction_analysis(self) -> Dict[str, Any]:
        """Get analysis of human-in-the-loop interactions."""
        if not self.human_interactions:
            return {"total_interactions": 0}
        
        # Analyze interaction types
        interaction_types = {}
        for interaction in self.human_interactions:
            itype = interaction['type']
            interaction_types[itype] = interaction_types.get(itype, 0) + 1
        
        return {
            "total_interactions": len(self.human_interactions),
            "interaction_types": interaction_types,
            "recent_interactions": [
                {
                    "type": i['type'],
                    "timestamp": i['timestamp'].isoformat(),
                    "session_id": i['session_id']
                }
                for i in self.human_interactions[-10:]  # Last 10 interactions
            ]
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
    
    def intercept_event_stream(self, stream_method: Callable, graph_instance: Any, method_name: str):
        """Intercept event streaming methods for real-time monitoring."""
        @functools.wraps(stream_method)
        def intercepted_stream(*args, **kwargs):
            session_id = f"stream_{id(graph_instance)}_{datetime.now().timestamp()}"
            
            # Track streaming session
            self.streaming_sessions[session_id] = {
                'start_time': datetime.now(),
                'graph_id': id(graph_instance),
                'method': method_name,
                'status': 'active'
            }
            
            self.logger.log_system_event(f"Started event streaming session: {session_id}")
            
            try:
                # Get the original stream
                stream = stream_method(*args, **kwargs)
                
                # Wrap the stream to monitor events
                def monitored_stream():
                    try:
                        for event in stream:
                            # Log and monitor each event
                            self._process_stream_event(event, session_id)
                            yield event
                    except Exception as e:
                        self.streaming_sessions[session_id]['status'] = 'error'
                        self.streaming_sessions[session_id]['error'] = str(e)
                        self.logger.log_system_event(f"Stream error in {session_id}: {e}")
                        raise
                    finally:
                        self.streaming_sessions[session_id]['status'] = 'completed'
                        self.streaming_sessions[session_id]['end_time'] = datetime.now()
                
                return monitored_stream()
                
            except Exception as e:
                self.streaming_sessions[session_id]['status'] = 'error'
                self.streaming_sessions[session_id]['error'] = str(e)
                raise
        
        return intercepted_stream
    
    async def intercept_async_event_stream(self, stream_method: Callable, graph_instance: Any, method_name: str):
        """Intercept async event streaming methods for real-time monitoring."""
        @functools.wraps(stream_method)
        async def intercepted_astream(*args, **kwargs):
            session_id = f"astream_{id(graph_instance)}_{datetime.now().timestamp()}"
            
            # Track streaming session
            self.streaming_sessions[session_id] = {
                'start_time': datetime.now(),
                'graph_id': id(graph_instance),
                'method': method_name,
                'status': 'active'
            }
            
            self.logger.log_system_event(f"Started async event streaming session: {session_id}")
            
            try:
                # Get the original stream
                stream = stream_method(*args, **kwargs)
                
                # Wrap the stream to monitor events
                async def monitored_astream():
                    try:
                        async for event in stream:
                            # Log and monitor each event
                            self._process_stream_event(event, session_id)
                            yield event
                    except Exception as e:
                        self.streaming_sessions[session_id]['status'] = 'error'
                        self.streaming_sessions[session_id]['error'] = str(e)
                        self.logger.log_system_event(f"Async stream error in {session_id}: {e}")
                        raise
                    finally:
                        self.streaming_sessions[session_id]['status'] = 'completed'
                        self.streaming_sessions[session_id]['end_time'] = datetime.now()
                
                return monitored_astream()
                
            except Exception as e:
                self.streaming_sessions[session_id]['status'] = 'error'
                self.streaming_sessions[session_id]['error'] = str(e)
                raise
        
        return intercepted_astream
    
    def _process_stream_event(self, event: dict, session_id: str):
        """Process and log streaming events."""
        try:
            event_type = event.get('event', 'unknown')
            event_name = event.get('name', 'unknown')
            
            # Log significant events
            if event_type in ['on_chain_start', 'on_chain_end', 'on_chain_error', 
                             'on_tool_start', 'on_tool_end', 'on_tool_error',
                             'on_llm_start', 'on_llm_end', 'on_llm_error']:
                
                event_record = {
                    'session_id': session_id,
                    'timestamp': datetime.now(),
                    'event_type': event_type,
                    'event_name': event_name,
                    'data': event.get('data', {}),
                }
                
                # Store event (keep last 1000 events)
                self.event_history.append(event_record)
                if len(self.event_history) > 1000:
                    self.event_history = self.event_history[-1000:]
                
                # Log detailed information for errors
                if 'error' in event_type:
                    self.logger.log_system_event(
                        f"Stream event error: {event_type} in {event_name}",
                        {"session_id": session_id, "event": event}
                    )
                else:
                    self.logger.log_system_event(
                        f"Stream event: {event_type} in {event_name}",
                        {"session_id": session_id}
                    )
                    
        except Exception as e:
            self.logger.log_system_event(f"Error processing stream event: {e}")
    
    def intercept_checkpoint_operation(self, checkpoint_method: Callable, checkpointer_instance: Any, method_name: str):
        """Intercept checkpoint operations for monitoring state persistence."""
        @functools.wraps(checkpoint_method)
        def intercepted_checkpoint(*args, **kwargs):
            operation_id = f"checkpoint_{method_name}_{datetime.now().timestamp()}"
            
            operation_record = {
                'operation_id': operation_id,
                'method': method_name,
                'checkpointer_type': type(checkpointer_instance).__name__,
                'start_time': datetime.now(),
                'args_summary': str(args)[:200] if args else None,
                'kwargs_summary': {k: str(v)[:100] for k, v in kwargs.items()},
            }
            
            try:
                result = checkpoint_method(*args, **kwargs)
                operation_record.update({
                    'status': 'success',
                    'end_time': datetime.now(),
                    'result_type': type(result).__name__ if result else None
                })
                
                self.logger.log_system_event(f"Checkpoint operation {method_name} completed successfully")
                
                return result
                
            except Exception as e:
                operation_record.update({
                    'status': 'error',
                    'end_time': datetime.now(),
                    'error': str(e)
                })
                
                self.logger.log_system_event(f"Checkpoint operation {method_name} failed: {e}")
                raise
                
            finally:
                # Store operation record
                self.checkpoint_operations.append(operation_record)
                if len(self.checkpoint_operations) > 500:  # Keep last 500 operations
                    self.checkpoint_operations = self.checkpoint_operations[-500:]
        
        return intercepted_checkpoint
    
    def intercept_create_react_agent(self, create_func: Callable):
        """Intercept create_react_agent function calls."""
        @functools.wraps(create_func)
        def intercepted_create_react_agent(*args, **kwargs):
            self.logger.log_system_event("Creating ReAct agent", {
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            })
            
            try:
                # Create the agent
                agent = create_func(*args, **kwargs)
                
                # Intercept the agent's methods
                if hasattr(agent, 'invoke'):
                    agent.invoke = self._create_sync_patched_method(
                        agent.invoke, 'ReActAgent', 'invoke'
                    )
                
                if hasattr(agent, 'ainvoke'):
                    agent.ainvoke = self._create_async_patched_method(
                        agent.ainvoke, 'ReActAgent', 'ainvoke'
                    )
                
                if hasattr(agent, 'stream'):
                    agent.stream = self.intercept_event_stream(
                        agent.stream, agent, 'stream'
                    )
                
                if hasattr(agent, 'astream'):
                    agent.astream = self.intercept_async_event_stream(
                        agent.astream, agent, 'astream'
                    )
                
                self.logger.log_system_event("Successfully created and intercepted ReAct agent")
                return agent
                
            except Exception as e:
                self.logger.log_system_event(f"Failed to create ReAct agent: {e}")
                raise
        
        return intercepted_create_react_agent
    
    def track_human_interaction(self, interaction_type: str, interaction_data: dict):
        """Track human-in-the-loop interactions."""
        interaction_record = {
            'timestamp': datetime.now(),
            'type': interaction_type,
            'data': interaction_data,
            'session_id': interaction_data.get('session_id', 'unknown')
        }
        
        self.human_interactions.append(interaction_record)
        if len(self.human_interactions) > 100:  # Keep last 100 interactions
            self.human_interactions = self.human_interactions[-100:]
        
        self.logger.log_system_event(
            f"Human interaction: {interaction_type}",
            {"session_id": interaction_record['session_id']}
        )
