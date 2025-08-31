"""
Utility decorators for Aigie monitoring.
"""

import functools
import asyncio
from typing import Optional, Callable, Any
from contextlib import contextmanager

from ..core.error_detector import ErrorDetector
from ..core.error_types import ErrorContext
from ..utils.config import AigieConfig


def monitor_execution(framework: str = "unknown", 
                     component: str = None, 
                     method: str = None,
                     config: Optional[AigieConfig] = None):
    """
    Decorator to monitor execution of a function or method.
    
    Args:
        framework: The framework being used (e.g., "langchain", "langgraph")
        component: The component name (defaults to function name)
        method: The method name (defaults to function name)
        config: Optional configuration override
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get component and method names
            comp_name = component or func.__name__
            meth_name = method or func.__name__
            
            # Create error context
            context = ErrorContext(
                timestamp=None,  # Will be set by monitor_execution
                framework=framework,
                component=comp_name,
                method=meth_name,
                input_data=_extract_input_data(args, kwargs)
            )
            
            # Get or create error detector
            error_detector = _get_error_detector(config)
            
            # Monitor execution
            with error_detector.monitor_execution(
                framework=framework,
                component=comp_name,
                method=meth_name,
                input_data=context.input_data
            ):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Error will be detected by the context manager
                    raise
        
        return wrapper
    return decorator


def monitor_async_execution(framework: str = "unknown",
                           component: str = None,
                           method: str = None,
                           config: Optional[AigieConfig] = None):
    """
    Decorator to monitor execution of an async function or method.
    
    Args:
        framework: The framework being used (e.g., "langchain", "langgraph")
        component: The component name (defaults to function name)
        method: The method name (defaults to function name)
        config: Optional configuration override
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get component and method names
            comp_name = component or func.__name__
            meth_name = method or func.__name__
            
            # Create error context
            context = ErrorContext(
                timestamp=None,  # Will be set by monitor_execution_async
                framework=framework,
                component=comp_name,
                method=meth_name,
                input_data=_extract_input_data(args, kwargs)
            )
            
            # Get or create error detector
            error_detector = _get_error_detector(config)
            
            # Monitor execution
            async with error_detector.monitor_execution_async(
                framework=framework,
                component=comp_name,
                method=meth_name,
                input_data=context.input_data
            ):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Error will be detected by the context manager
                    raise
        
        return wrapper
    return decorator


def monitor_langchain(component: str = None, method: str = None):
    """Decorator specifically for LangChain operations."""
    return monitor_execution(framework="langchain", component=component, method=method)


def monitor_langgraph(component: str = None, method: str = None):
    """Decorator specifically for LangGraph operations."""
    return monitor_execution(framework="langgraph", component=component, method=method)


def monitor_langchain_async(component: str = None, method: str = None):
    """Decorator specifically for async LangChain operations."""
    return monitor_async_execution(framework="langchain", component=component, method=method)


def monitor_langgraph_async(component: str = None, method: str = None):
    """Decorator specifically for async LangGraph operations."""
    return monitor_async_execution(framework="langgraph", component=component, method=method)


def _extract_input_data(args: tuple, kwargs: dict) -> Optional[dict]:
    """Extract relevant input data for monitoring."""
    input_data = {}
    
    # Extract first few arguments (truncate long ones)
    for i, arg in enumerate(args[:3]):  # Limit to first 3 args
        try:
            arg_str = str(arg)
            if len(arg_str) > 200:
                arg_str = arg_str[:200] + "..."
            input_data[f"arg_{i}"] = arg_str
        except Exception:
            input_data[f"arg_{i}"] = f"<{type(arg).__name__}>"
    
    # Extract keyword arguments (truncate long ones)
    for key, value in kwargs.items():
        if key in ['input', 'inputs', 'query', 'text', 'prompt', 'state']:
            try:
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                input_data[key] = value_str
            except Exception:
                input_data[key] = f"<{type(value).__name__}>"
        elif key in ['config', 'callbacks', 'memory', 'tools']:
            input_data[key] = type(value).__name__
    
    return input_data if input_data else None


def _get_error_detector(config: Optional[AigieConfig] = None) -> ErrorDetector:
    """Get or create an error detector instance."""
    # Try to get from global integrator first
    try:
        from ..auto_integration import get_integrator
        integrator = get_integrator()
        if integrator and integrator.error_detector.is_monitoring:
            return integrator.error_detector
    except ImportError:
        pass
    
    # Create a new error detector if needed
    if config is None:
        config = AigieConfig()
    
    return ErrorDetector(
        enable_performance_monitoring=config.enable_performance_monitoring,
        enable_resource_monitoring=config.enable_resource_monitoring
    )


# Context managers for manual monitoring
@contextmanager
def monitor_function_call(framework: str = "unknown",
                         component: str = "unknown",
                         method: str = "unknown",
                         config: Optional[AigieConfig] = None):
    """Context manager for monitoring function calls."""
    error_detector = _get_error_detector(config)
    
    with error_detector.monitor_execution(
        framework=framework,
        component=component,
        method=method
    ):
        yield


@contextmanager
def monitor_langchain_call(component: str = "unknown", method: str = "unknown"):
    """Context manager for monitoring LangChain calls."""
    with monitor_function_call(framework="langchain", component=component, method=method):
        yield


@contextmanager
def monitor_langgraph_call(component: str = "unknown", method: str = "unknown"):
    """Context manager for monitoring LangGraph calls."""
    with monitor_function_call(framework="langgraph", component=component, method=method):
        yield


# Performance monitoring decorators
def track_performance(threshold_seconds: float = 30.0):
    """Decorator to track performance and warn if execution is slow."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                if execution_time > threshold_seconds:
                    print(f"⚠️  Slow execution detected: {func.__name__} took {execution_time:.2f}s (threshold: {threshold_seconds}s)")
        
        return wrapper
    return decorator


def track_memory_usage():
    """Decorator to track memory usage of a function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import psutil
            import gc
            
            # Force garbage collection before
            gc.collect()
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Force garbage collection after
                gc.collect()
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = memory_after - memory_before
                
                if abs(memory_delta) > 10:  # Warn if memory change > 10MB
                    direction = "increased" if memory_delta > 0 else "decreased"
                    print(f"💾 Memory {direction}: {func.__name__} {direction} memory by {abs(memory_delta):.2f}MB")
        
        return wrapper
    return decorator


# Combined monitoring decorator
def comprehensive_monitoring(framework: str = "unknown",
                           component: str = None,
                           method: str = None,
                           performance_threshold: float = 30.0,
                           track_memory: bool = True):
    """Comprehensive monitoring decorator that combines multiple monitoring aspects."""
    def decorator(func: Callable) -> Callable:
        # Apply performance monitoring
        func = track_performance(performance_threshold)(func)
        
        # Apply memory tracking if requested
        if track_memory:
            func = track_memory_usage()(func)
        
        # Apply error detection monitoring
        if asyncio.iscoroutinefunction(func):
            func = monitor_async_execution(framework, component, method)(func)
        else:
            func = monitor_execution(framework, component, method)(func)
        
        return func
    return decorator


# LangChain specific comprehensive monitoring
def monitor_langchain_comprehensive(component: str = None,
                                   method: str = None,
                                   performance_threshold: float = 30.0,
                                   track_memory: bool = True):
    """Comprehensive monitoring for LangChain operations."""
    return comprehensive_monitoring(
        framework="langchain",
        component=component,
        method=method,
        performance_threshold=performance_threshold,
        track_memory=track_memory
    )


# LangGraph specific comprehensive monitoring
def monitor_langgraph_comprehensive(component: str = None,
                                   method: str = None,
                                   performance_threshold: float = 30.0,
                                   track_memory: bool = True):
    """Comprehensive monitoring for LangGraph operations."""
    return comprehensive_monitoring(
        framework="langgraph",
        component=component,
        method=method,
        performance_threshold=performance_threshold,
        track_memory=track_memory
    )
