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
        
        # LangChain components to intercept (updated for modern LangChain)
        self.target_classes = {
            # Modern Chat Models (primary LLM interface)
            'ChatOpenAI': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            'ChatAnthropic': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            'ChatGoogleGenerativeAI': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            'ChatOllama': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            
            # Legacy LLM support (still used in some cases)
            'OpenAI': ['invoke', 'ainvoke', 'batch', 'abatch', '__call__', 'acall', 'agenerate', 'generate'],
            'LLM': ['invoke', 'ainvoke', 'batch', 'abatch', '__call__', 'acall', 'agenerate', 'generate'],
            
            # Modern Tool System
            'BaseTool': ['invoke', 'ainvoke', '_run', '_arun', 'run', 'arun'],
            'StructuredTool': ['invoke', 'ainvoke', '_run', '_arun', 'run', 'arun'],
            
            # LCEL Runnable Components (core of modern LangChain)
            'RunnablePassthrough': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            'RunnableLambda': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            'RunnableParallel': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            'RunnableSequence': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            'RunnableBranch': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            
            # Output Parsers
            'BaseOutputParser': ['parse', 'aparse', 'parse_result', 'aparse_result', 'invoke', 'ainvoke'],
            'StrOutputParser': ['parse', 'aparse', 'parse_result', 'aparse_result', 'invoke', 'ainvoke'],
            'PydanticOutputParser': ['parse', 'aparse', 'parse_result', 'aparse_result', 'invoke', 'ainvoke'],
            
            # Retrieval Components
            'BaseRetriever': ['invoke', 'ainvoke', 'get_relevant_documents', 'aget_relevant_documents'],
            'VectorStoreRetriever': ['invoke', 'ainvoke', 'get_relevant_documents', 'aget_relevant_documents'],
            
            # Agents (modern agent system)
            'AgentExecutor': ['invoke', 'ainvoke', 'batch', 'abatch', 'stream', 'astream'],
            
            # Legacy support (for backwards compatibility)
            'LLMChain': ['invoke', 'ainvoke', 'run', '__call__', 'acall', 'arun'],
            'Agent': ['run', '__call__', 'acall', 'arun'],
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
        classes_to_patch = {}
        
        # Modern Chat Models
        try:
            from langchain_openai import ChatOpenAI
            classes_to_patch['ChatOpenAI'] = ChatOpenAI
        except Exception as e:
            self.logger.log_system_event(f"ChatOpenAI not available: {e}")
        
        try:
            from langchain_anthropic import ChatAnthropic
            classes_to_patch['ChatAnthropic'] = ChatAnthropic
        except Exception as e:
            self.logger.log_system_event(f"ChatAnthropic not available: {e}")
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            classes_to_patch['ChatGoogleGenerativeAI'] = ChatGoogleGenerativeAI
        except Exception as e:
            self.logger.log_system_event(f"ChatGoogleGenerativeAI not available: {e}")
        
        try:
            from langchain_community.chat_models import ChatOllama
            classes_to_patch['ChatOllama'] = ChatOllama
        except Exception as e:
            self.logger.log_system_event(f"ChatOllama not available: {e}")
        
        # Legacy LLMs
        try:
            from langchain_openai import OpenAI
            classes_to_patch['OpenAI'] = OpenAI
        except Exception as e:
            try:
                from langchain.llms import OpenAI
                classes_to_patch['OpenAI'] = OpenAI
            except Exception as e2:
                self.logger.log_system_event(f"OpenAI LLM not available: {e2}")
        
        try:
            from langchain.llms.base import LLM
            classes_to_patch['LLM'] = LLM
        except Exception as e:
            self.logger.log_system_event(f"Base LLM not available: {e}")
        
        # Modern Tool System
        try:
            from langchain_core.tools import BaseTool, StructuredTool
            classes_to_patch['BaseTool'] = BaseTool
            classes_to_patch['StructuredTool'] = StructuredTool
        except Exception as e:
            try:
                from langchain.tools import BaseTool, StructuredTool
                classes_to_patch['BaseTool'] = BaseTool
                classes_to_patch['StructuredTool'] = StructuredTool
            except Exception as e2:
                self.logger.log_system_event(f"Modern tool classes not available: {e2}")
        
        # LCEL Runnable Components
        try:
            from langchain_core.runnables import (
                RunnablePassthrough, RunnableLambda, RunnableParallel, 
                RunnableSequence, RunnableBranch
            )
            classes_to_patch.update({
                'RunnablePassthrough': RunnablePassthrough,
                'RunnableLambda': RunnableLambda,
                'RunnableParallel': RunnableParallel,
                'RunnableSequence': RunnableSequence,
                'RunnableBranch': RunnableBranch,
            })
        except Exception as e:
            self.logger.log_system_event(f"LCEL Runnable components not available: {e}")
        
        # Output Parsers
        try:
            from langchain_core.output_parsers import BaseOutputParser, StrOutputParser, PydanticOutputParser
            classes_to_patch.update({
                'BaseOutputParser': BaseOutputParser,
                'StrOutputParser': StrOutputParser,
                'PydanticOutputParser': PydanticOutputParser,
            })
        except Exception as e:
            try:
                from langchain.output_parsers import BaseOutputParser, StrOutputParser, PydanticOutputParser
                classes_to_patch.update({
                    'BaseOutputParser': BaseOutputParser,
                    'StrOutputParser': StrOutputParser,
                    'PydanticOutputParser': PydanticOutputParser,
                })
            except Exception as e2:
                self.logger.log_system_event(f"Output parsers not available: {e2}")
        
        # Retrieval Components
        try:
            from langchain_core.retrievers import BaseRetriever
            from langchain.vectorstores.base import VectorStoreRetriever
            classes_to_patch.update({
                'BaseRetriever': BaseRetriever,
                'VectorStoreRetriever': VectorStoreRetriever,
            })
        except Exception as e:
            self.logger.log_system_event(f"Retrieval components not available: {e}")
        
        # Modern Agent System
        try:
            from langchain.agents import AgentExecutor
            classes_to_patch['AgentExecutor'] = AgentExecutor
        except Exception as e:
            self.logger.log_system_event(f"AgentExecutor not available: {e}")
        
        # Legacy support
        try:
            from langchain.chains import LLMChain
            from langchain.agents import Agent
            classes_to_patch.update({
                'LLMChain': LLMChain,
                'Agent': Agent,
            })
        except Exception as e:
            self.logger.log_system_event(f"Legacy LangChain classes not available: {e}")
        
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
                # Get the method descriptor from the class
                method_descriptor = getattr(cls, method_name)
                
                # Store original method descriptor
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
                    framework="langchain",
                    component=class_name,
                    method=method_name,
                    input_data=self._extract_input_data(args, kwargs, method_name)
                )
                
                # Store operation for potential retry
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + args, kwargs, context
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
        else:
            def patched_method(self_instance, *args, **kwargs):
                # Create error context
                context = ErrorContext(
                    timestamp=datetime.now(),
                    framework="langchain",
                    component=class_name,
                    method=method_name,
                    input_data=self._extract_input_data(args, kwargs, method_name)
                )
                
                # Store operation for potential retry
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + args, kwargs, context
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
        # Only use functools.wraps if original_method is actually callable and has the required attributes
        if callable(original_method) and hasattr(original_method, '__name__'):
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
                
                # Store operation for potential retry
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + args, kwargs, context
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
        else:
            async def patched_method(self_instance, *args, **kwargs):
                # Create error context
                context = ErrorContext(
                    timestamp=datetime.now(),
                    framework="langchain",
                    component=class_name,
                    method=method_name,
                    input_data=self._extract_input_data(args, kwargs, method_name)
                )
                
                # Store operation for potential retry
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + args, kwargs, context
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
                    if hasattr(cls, '__name__') and cls.__name__ == class_name:
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
    
    def intercept_chain(self, chain_instance: Any):
        """Intercept a specific chain instance."""
        class_name = chain_instance.__class__.__name__
        
        if class_name in self.target_classes:
            methods_to_patch = self.target_classes[class_name]
            
            for method_name in methods_to_patch:
                if hasattr(chain_instance, method_name):
                    method_descriptor = getattr(chain_instance, method_name)
                    
                    # Debug logging to see what we're storing
                    self.logger.log_system_event(f"Storing instance method {chain_instance.__class__.__name__}.{method_name}: type={type(method_descriptor)}, callable={callable(method_descriptor)}")
                    
                    # Store original method
                    key = f"{chain_instance.__class__.__name__}.{method_name}"
                    self.original_methods[key] = method_descriptor
                    
                    # Create patched method
                    if inspect.iscoroutinefunction(method_descriptor):
                        patched_method = self._create_async_patched_method(method_descriptor, class_name, method_name)
                    else:
                        patched_method = self._create_sync_patched_method(method_descriptor, class_name, method_name)
                    
                    # Apply the patch
                    setattr(chain_instance, method_name, patched_method)
            
            self.logger.log_system_event(f"Intercepted chain instance: {class_name}")
    
    def intercept_tool_function(self, tool_func: Callable, tool_name: str = None) -> Callable:
        """Intercept a tool function decorated with @tool."""
        tool_name = tool_name or getattr(tool_func, '__name__', 'unknown_tool')
        
        @functools.wraps(tool_func)
        def intercepted_tool(*args, **kwargs):
            # Create error context
            context = ErrorContext(
                timestamp=datetime.now(),
                framework="langchain",
                component="Tool",
                method="__call__",
                input_data=self._extract_tool_input_data(args, kwargs, tool_name)
            )
            
            # Store operation for potential retry
            operation_id = f"langchain_Tool_{tool_name}"
            self.error_detector.store_operation_for_retry(
                operation_id, tool_func, args, kwargs, context
            )
            
            # Monitor execution
            with self.error_detector.monitor_execution(
                framework="langchain",
                component="Tool",
                method="__call__",
                input_data=context.input_data
            ):
                try:
                    result = tool_func(*args, **kwargs)
                    self.logger.log_system_event(f"Tool '{tool_name}' executed successfully")
                    return result
                except Exception as e:
                    self.logger.log_system_event(f"Tool '{tool_name}' execution failed: {e}")
                    raise
        
        return intercepted_tool
    
    def _extract_tool_input_data(self, args: tuple, kwargs: dict, tool_name: str) -> Optional[Dict[str, Any]]:
        """Extract input data from tool function call."""
        input_data = {"tool_name": tool_name}
        
        # Add positional args (usually tool inputs)
        if args:
            input_data["args"] = [str(arg)[:100] for arg in args[:3]]  # First 3 args, truncated
        
        # Add keyword arguments
        for key, value in kwargs.items():
            if key in ['input', 'query', 'text', 'prompt']:
                input_data[key] = str(value)[:200]  # Truncate long inputs
            else:
                input_data[key] = type(value).__name__
        
        return input_data
    
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
