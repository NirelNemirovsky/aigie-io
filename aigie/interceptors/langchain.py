"""
LangChain interceptor for real-time error detection and monitoring.
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime

from ..core.error_detector import ErrorDetector
from ..core.error_types import ErrorContext
from ..reporting.logger import AigieLogger


class LangChainInterceptor:
    """Intercepts LangChain operations to detect errors and monitor performance."""
    
    def __init__(self, error_detector: ErrorDetector, logger: AigieLogger):
        self.error_detector = error_detector
        self.logger = logger
        self.intercepted_classes = set()
        self.original_methods = {}
        
        # LangChain components to intercept
        self.target_classes = {
            'LLMChain': ['run', '__call__', 'acall', 'arun'],
            'Agent': ['run', '__call__', 'acall', 'arun'],
            'Tool': ['run', '__call__', 'acall', 'arun'],
            'LLM': ['__call__', 'acall', 'agenerate', 'generate'],
        }
    
    def start_intercepting(self):
        """Start intercepting LangChain operations."""
        self.error_detector.start_monitoring()
        self.logger.log_system_event("Started LangChain interception")
        
        # Intercept existing instances
        self._intercept_existing_instances()
        
        # Patch class methods for future instances
        self._patch_classes()
    
    def stop_intercepting(self):
        """Stop intercepting LangChain operations."""
        self.error_detector.stop_monitoring()
        self.logger.log_system_event("Stopped LangChain interception")
        
        # Restore original methods
        self._restore_original_methods()
    
    def _intercept_existing_instances(self):
        """Intercept existing LangChain instances."""
        # This would require access to a registry of instances
        # For now, we'll focus on patching classes for future instances
        pass
    
    def _patch_classes(self):
        """Patch LangChain classes to intercept method calls."""
        try:
            # Import LangChain classes dynamically
            self._patch_langchain_classes()
        except ImportError as e:
            self.logger.log_system_event(f"Could not import LangChain classes: {e}")
    
    def _patch_langchain_classes(self):
        """Patch specific LangChain classes."""
        # Try to import and patch LangChain classes
        try:
            # Try to import newer LangChain classes
            try:
                from langchain.chains import LLMChain
                from langchain.agents import Agent
                from langchain.tools import BaseTool
                from langchain.llms.base import LLM
                
                classes_to_patch = {
                    'LLMChain': LLMChain,
                    'Agent': Agent,
                    'Tool': BaseTool,
                    'LLM': LLM,
                }
                
                for class_name, cls in classes_to_patch.items():
                    if cls and class_name in self.target_classes:
                        self._patch_class_methods(cls, class_name)
                        
            except ImportError:
                # If newer classes aren't available, just log it
                self.logger.log_system_event("LangChain newer classes not available")
                
        except ImportError as e:
            self.logger.log_system_event(f"LangChain not available: {e}")
    
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
                framework="langchain",
                component=class_name,
                method=method_name,
                input_data=self._extract_input_data(args, kwargs, method_name)
            )
            
            # Monitor execution
            with self.error_detector.monitor_execution(
                framework="langchain",
                component=class_name,
                method=method_name,
                input_data=context.input_data
            ):
                try:
                    # Call original method
                    result = original_method(self_instance, *args, **kwargs)
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
                framework="langchain",
                component=class_name,
                method=method_name,
                input_data=self._extract_input_data(args, kwargs, method_name)
            )
            
            # Monitor execution
            async with self.error_detector.monitor_execution_async(
                framework="langchain",
                component=class_name,
                method=method_name,
                input_data=context.input_data
            ):
                try:
                    # Call original method
                    result = await original_method(self_instance, *args, **kwargs)
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
            if method_name in ['run', '__call__', 'acall', 'arun']:
                if args:
                    input_data['input'] = str(args[0])[:200]  # Truncate long inputs
        
        # Extract keyword arguments
        for key, value in kwargs.items():
            if key in ['input', 'inputs', 'query', 'text', 'prompt']:
                input_data[key] = str(value)[:200]  # Truncate long inputs
            elif key in ['memory', 'tools', 'callbacks']:
                input_data[key] = type(value).__name__  # Just the type
        
        return input_data if input_data else None
    
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
    
    def intercept_chain(self, chain_instance: Any):
        """Intercept a specific chain instance."""
        class_name = chain_instance.__class__.__name__
        
        if class_name in self.target_classes:
            methods_to_patch = self.target_classes[class_name]
            
            for method_name in methods_to_patch:
                if hasattr(chain_instance, method_name):
                    original_method = getattr(chain_instance, method_name)
                    
                    # Store original method
                    key = f"{chain_instance.__class__.__name__}.{method_name}"
                    self.original_methods[key] = original_method
                    
                    # Create patched method
                    if inspect.iscoroutinefunction(original_method):
                        patched_method = self._create_async_patched_method(original_method, class_name, method_name)
                    else:
                        patched_method = self._create_sync_patched_method(original_method, class_name, method_name)
                    
                    # Apply the patch
                    setattr(chain_instance, method_name, patched_method)
            
            self.logger.log_system_event(f"Intercepted chain instance: {class_name}")
    
    def get_interception_status(self) -> Dict[str, Any]:
        """Get current interception status."""
        return {
            "is_intercepting": self.error_detector.is_monitoring,
            "intercepted_classes": list(self.intercepted_classes),
            "patched_methods": list(self.original_methods.keys()),
            "target_classes": list(self.target_classes.keys())
        }


class LangChainCallbackHandler:
    """LangChain callback handler for integration with existing callback system."""
    
    def __init__(self, error_detector: ErrorDetector, logger: AigieLogger):
        self.error_detector = error_detector
        self.logger = logger
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when a chain starts."""
        if self.error_detector.is_monitoring:
            self.logger.log_system_event(
                "Chain started",
                {
                    "chain_name": serialized.get("name", "unknown"),
                    "inputs": str(inputs)[:200]
                }
            )
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when a chain ends."""
        if self.error_detector.is_monitoring:
            self.logger.log_system_event(
                "Chain completed",
                {"outputs": str(outputs)[:200]}
            )
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs):
        """Called when a chain encounters an error."""
        if self.error_detector.is_monitoring:
            # Create error context
            context = ErrorContext(
                timestamp=datetime.now(),
                framework="langchain",
                component="Chain",
                method="run",
                stack_trace=str(error)
            )
            
            # Let the error detector handle it
            self.error_detector._detect_error(error, context)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when a tool starts."""
        if self.error_detector.is_monitoring:
            self.logger.log_system_event(
                "Tool started",
                {
                    "tool_name": serialized.get("name", "unknown"),
                    "input": input_str[:200]
                }
            )
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when a tool ends."""
        if self.error_detector.is_monitoring:
            self.logger.log_system_event(
                "Tool completed",
                {"output": output[:200]}
            )
    
    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs):
        """Called when a tool encounters an error."""
        if self.error_detector.is_monitoring:
            # Create error context
            context = ErrorContext(
                timestamp=datetime.now(),
                framework="langchain",
                component="Tool",
                method="run",
                stack_trace=str(error)
            )
            
            # Let the error detector handle it
            self.error_detector._detect_error(error, context)
