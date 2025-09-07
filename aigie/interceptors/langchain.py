"""
LangChain interceptor for real-time error detection and monitoring.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime

from ..core.error_handling.error_detector import ErrorDetector
from ..core.types.error_types import ErrorContext
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
                
                # Apply intelligent prompt injection based on operation type
                enhanced_args, enhanced_kwargs = self._apply_smart_prompt_injection(
                    args, kwargs, class_name, method_name, self_instance
                )
                
                # Store operation for potential retry (with enhanced args/kwargs)
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + enhanced_args, enhanced_kwargs, context
                )
                
                # Monitor execution with retry capability
                try:
                    with self.error_detector.monitor_execution(
                        framework="langchain",
                        component=class_name,
                        method=method_name,
                        input_data=context.input_data
                    ):
                        # Call original method with potentially enhanced arguments
                        result = original_method(self_instance, *enhanced_args, **enhanced_kwargs)
                        return result
                except Exception as e:
                    # Try automatic retry if enabled
                    if (self.error_detector.enable_automatic_retry and 
                        self.error_detector.intelligent_retry):
                        try:
                            # Create error context for retry
                            error_context = ErrorContext(
                                timestamp=datetime.now(),
                                framework="langchain",
                                component=class_name,
                                method=method_name,
                                input_data=context.input_data
                            )
                            
                            # Attempt retry with enhanced context
                            
                            # Attempt retry with enhanced context
                            retry_result = self.error_detector.intelligent_retry.retry_with_gemini_context(
                                original_method, self_instance, *args, 
                                error_context=error_context, **kwargs
                            )
                            
                            if retry_result is not None:
                                logging.info(f"✅ LANGCHAIN RETRY SUCCESS: {class_name}.{method_name} recovered")
                                return retry_result
                        except Exception as retry_error:
                            logging.warning(f"LangChain retry failed: {retry_error}")
                    
                    # If retry failed or not enabled, raise original exception
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
                
                # Apply intelligent prompt injection based on operation type
                enhanced_args, enhanced_kwargs = self._apply_smart_prompt_injection(
                    args, kwargs, class_name, method_name, self_instance
                )
                
                # Store operation for potential retry (with enhanced args/kwargs)
                operation_id = f"{context.framework}_{context.component}_{context.method}"
                self.error_detector.store_operation_for_retry(
                    operation_id, original_method, (self_instance,) + enhanced_args, enhanced_kwargs, context
                )
                
                # Monitor execution with retry capability
                try:
                    with self.error_detector.monitor_execution(
                        framework="langchain",
                        component=class_name,
                        method=method_name,
                        input_data=context.input_data
                    ):
                        # Call original method with potentially enhanced arguments
                        result = original_method(self_instance, *enhanced_args, **enhanced_kwargs)
                        return result
                except Exception as e:
                    # Try automatic retry if enabled
                    if (self.error_detector.enable_automatic_retry and 
                        self.error_detector.intelligent_retry):
                        try:
                            # Create error context for retry
                            error_context = ErrorContext(
                                timestamp=datetime.now(),
                                framework="langchain",
                                component=class_name,
                                method=method_name,
                                input_data=context.input_data
                            )
                            
                            # Attempt retry with enhanced context
                            
                            # Attempt retry with enhanced context
                            retry_result = self.error_detector.intelligent_retry.retry_with_gemini_context(
                                original_method, self_instance, *args, 
                                error_context=error_context, **kwargs
                            )
                            
                            if retry_result is not None:
                                logging.info(f"✅ LANGCHAIN RETRY SUCCESS: {class_name}.{method_name} recovered")
                                return retry_result
                        except Exception as retry_error:
                            logging.warning(f"LangChain retry failed: {retry_error}")
                    
                    # If retry failed or not enabled, raise original exception
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
                
                # Monitor execution with retry capability
                try:
                    async with self.error_detector.monitor_execution_async(
                        framework="langchain",
                        component=class_name,
                        method=method_name,
                        input_data=context.input_data
                    ):
                        # Call original method
                        result = await original_method(self_instance, *args, **kwargs)
                        return result
                except Exception as e:
                    # Try automatic retry if enabled
                    if (self.error_detector.enable_automatic_retry and 
                        self.error_detector.intelligent_retry):
                        try:
                            # Create error context for retry
                            error_context = ErrorContext(
                                timestamp=datetime.now(),
                                framework="langchain",
                                component=class_name,
                                method=method_name,
                                input_data=context.input_data
                            )
                            
                            # Attempt retry with enhanced context
                            
                            # Attempt retry with enhanced context
                            retry_result = self.error_detector.intelligent_retry.retry_with_gemini_context(
                                original_method, self_instance, *args, 
                                error_context=error_context, **kwargs
                            )
                            
                            if retry_result is not None:
                                logging.info(f"✅ LANGCHAIN ASYNC RETRY SUCCESS: {class_name}.{method_name} recovered")
                                return retry_result
                        except Exception as retry_error:
                            logging.warning(f"LangChain async retry failed: {retry_error}")
                    
                    # If retry failed or not enabled, raise original exception
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
                
                # Monitor execution with retry capability
                try:
                    async with self.error_detector.monitor_execution_async(
                        framework="langchain",
                        component=class_name,
                        method=method_name,
                        input_data=context.input_data
                    ):
                        # Call original method
                        result = await original_method(self_instance, *args, **kwargs)
                        return result
                except Exception as e:
                    # Try automatic retry if enabled
                    if (self.error_detector.enable_automatic_retry and 
                        self.error_detector.intelligent_retry):
                        try:
                            # Create error context for retry
                            error_context = ErrorContext(
                                timestamp=datetime.now(),
                                framework="langchain",
                                component=class_name,
                                method=method_name,
                                input_data=context.input_data
                            )
                            
                            # Attempt retry with enhanced context
                            
                            # Attempt retry with enhanced context
                            retry_result = self.error_detector.intelligent_retry.retry_with_gemini_context(
                                original_method, self_instance, *args, 
                                error_context=error_context, **kwargs
                            )
                            
                            if retry_result is not None:
                                logging.info(f"✅ LANGCHAIN ASYNC RETRY SUCCESS: {class_name}.{method_name} recovered")
                                return retry_result
                        except Exception as retry_error:
                            logging.warning(f"LangChain async retry failed: {retry_error}")
                    
                    # If retry failed or not enabled, raise original exception
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
            
            # Monitor execution with retry capability
            try:
                with self.error_detector.monitor_execution(
                    framework="langchain",
                    component="Tool",
                    method="__call__",
                    input_data=context.input_data
                ):
                    result = tool_func(*args, **kwargs)
                    self.logger.log_system_event(f"Tool '{tool_name}' executed successfully")
                    return result
            except Exception as e:
                self.logger.log_system_event(f"Tool '{tool_name}' execution failed: {e}")
                
                # Try automatic retry if enabled
                if (self.error_detector.enable_automatic_retry and 
                    self.error_detector.intelligent_retry):
                    try:
                        # Create error context for retry
                        error_context = ErrorContext(
                            timestamp=datetime.now(),
                            framework="langchain",
                            component="Tool",
                            method="__call__",
                            input_data=context.input_data
                        )
                        
                        # Attempt retry with enhanced context
                        retry_result = self.error_detector.intelligent_retry.retry_with_gemini_context(
                            tool_func, *args, 
                            error_context=error_context, **kwargs
                        )
                        
                        if retry_result is not None:
                            logging.info(f"✅ TOOL RETRY SUCCESS: {tool_name} recovered")
                            return retry_result
                    except Exception as retry_error:
                        logging.warning(f"Tool retry failed: {retry_error}")
                
                # If retry failed or not enabled, raise original exception
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
            "intercepted_classes": [cls.__name__ if hasattr(cls, '__name__') else str(cls) for cls in self.intercepted_classes],
            "patched_methods": list(self.original_methods.keys()),
            "target_classes": list(self.target_classes.keys())
        }
    
    def _is_llm_call(self, class_name: str, method_name: str) -> bool:
        """Check if this is an LLM call that should receive prompt injection."""
        # Only target actual LLM classes, not agents or chains
        llm_classes = {
            'ChatOpenAI', 'ChatAnthropic', 'ChatGoogleGenerativeAI', 'ChatOllama',
            'OpenAI', 'LLM', 'ChatModel', 'BaseLanguageModel'
        }
        
        # Exclude agent and chain classes
        excluded_classes = {
            'Agent', 'AgentExecutor', 'LLMChain', 'ConversationChain',
            'RetrievalQA', 'VectorStoreRetriever', 'BaseRetriever'
        }
        
        # Methods that represent direct LLM calls
        llm_methods = {'invoke', 'ainvoke', '__call__', 'acall', 'generate', 'agenerate'}
        
        # Check if it's an LLM class and not an excluded class
        is_llm_class = class_name in llm_classes
        is_excluded = any(excluded in class_name for excluded in excluded_classes)
        is_llm_method = method_name in llm_methods
        
        return is_llm_class and not is_excluded and is_llm_method
    
    def _apply_smart_prompt_injection(self, args: tuple, kwargs: dict, class_name: str, 
                                    method_name: str, instance: Any) -> tuple:
        """Apply intelligent prompt injection based on operation type and context."""
        enhanced_args = list(args)
        enhanced_kwargs = kwargs.copy()
        
        # Check if we have pending remediation prompts
        if (hasattr(self.error_detector, 'pending_remediation_prompts') and 
            self.error_detector.pending_remediation_prompts):
            
            remediation_prompt = self.error_detector.pending_remediation_prompts[0]
            logging.info(f"💉 SMART INJECTION: Applying remediation prompt for {class_name}.{method_name}")
            
            # Determine the best injection strategy based on operation type
            if self._is_agent_operation(class_name, method_name):
                # For agent operations, inject via input parameter
                enhanced_kwargs = self._inject_agent_prompt(enhanced_kwargs, remediation_prompt)
            elif self._is_llm_operation(class_name, method_name):
                # For LLM operations, inject via prompt parameter
                enhanced_kwargs = self._inject_llm_prompt(enhanced_kwargs, remediation_prompt)
            elif self._is_chain_operation(class_name, method_name):
                # For chain operations, inject via input parameter
                enhanced_kwargs = self._inject_chain_prompt(enhanced_kwargs, remediation_prompt)
            else:
                # Generic injection
                enhanced_kwargs = self._inject_generic_prompt(enhanced_kwargs, remediation_prompt)
            
            # Clear the used remediation prompt
            self.error_detector.pending_remediation_prompts.clear()
        
        return tuple(enhanced_args), enhanced_kwargs
    
    def _is_agent_operation(self, class_name: str, method_name: str) -> bool:
        """Check if this is an agent operation."""
        agent_classes = ['AgentExecutor', 'Agent', 'ConversationalAgent', 'ReActAgent']
        return any(agent_class in class_name for agent_class in agent_classes)
    
    def _is_llm_operation(self, class_name: str, method_name: str) -> bool:
        """Check if this is an LLM operation."""
        llm_classes = ['OpenAI', 'ChatOpenAI', 'Anthropic', 'ChatAnthropic', 'LLM', 'ChatModel']
        return any(llm_class in class_name for llm_class in llm_classes)
    
    def _is_chain_operation(self, class_name: str, method_name: str) -> bool:
        """Check if this is a chain operation."""
        chain_classes = ['Chain', 'LLMChain', 'ConversationChain', 'RetrievalQA']
        return any(chain_class in class_name for chain_class in chain_classes)
    
    def _inject_agent_prompt(self, kwargs: dict, remediation_prompt: str) -> dict:
        """Inject prompt for agent operations."""
        enhanced_kwargs = kwargs.copy()
        
        # For agents, inject via input parameter
        if 'input' in enhanced_kwargs:
            original_input = enhanced_kwargs['input']
            enhanced_kwargs['input'] = f"{remediation_prompt}\n\n{original_input}"
            logging.info("💉 AGENT INJECTION: Enhanced input parameter")
        elif 'inputs' in enhanced_kwargs:
            original_inputs = enhanced_kwargs['inputs']
            if isinstance(original_inputs, dict) and 'input' in original_inputs:
                original_inputs['input'] = f"{remediation_prompt}\n\n{original_inputs['input']}"
                logging.info("💉 AGENT INJECTION: Enhanced inputs['input'] parameter")
        
        return enhanced_kwargs
    
    def _inject_llm_prompt(self, kwargs: dict, remediation_prompt: str) -> dict:
        """Inject prompt for LLM operations."""
        enhanced_kwargs = kwargs.copy()
        
        # For LLMs, inject via messages or prompt parameter
        if 'messages' in enhanced_kwargs and isinstance(enhanced_kwargs['messages'], list):
            messages = enhanced_kwargs['messages'].copy()
            system_message = {
                "role": "system",
                "content": f"{remediation_prompt}\n\nYou are an AI assistant. Please follow the guidance above when responding."
            }
            messages.insert(0, system_message)
            enhanced_kwargs['messages'] = messages
            logging.info("💉 LLM INJECTION: Enhanced messages with system prompt")
        elif 'prompt' in enhanced_kwargs:
            original_prompt = enhanced_kwargs['prompt']
            enhanced_kwargs['prompt'] = f"{remediation_prompt}\n\n{original_prompt}"
            logging.info("💉 LLM INJECTION: Enhanced prompt parameter")
        
        return enhanced_kwargs
    
    def _inject_chain_prompt(self, kwargs: dict, remediation_prompt: str) -> dict:
        """Inject prompt for chain operations."""
        enhanced_kwargs = kwargs.copy()
        
        # For chains, inject via input parameter
        if 'input' in enhanced_kwargs:
            original_input = enhanced_kwargs['input']
            enhanced_kwargs['input'] = f"{remediation_prompt}\n\n{original_input}"
            logging.info("💉 CHAIN INJECTION: Enhanced input parameter")
        elif 'inputs' in enhanced_kwargs:
            original_inputs = enhanced_kwargs['inputs']
            if isinstance(original_inputs, dict) and 'input' in original_inputs:
                original_inputs['input'] = f"{remediation_prompt}\n\n{original_inputs['input']}"
                logging.info("💉 CHAIN INJECTION: Enhanced inputs['input'] parameter")
        
        return enhanced_kwargs
    
    def _inject_generic_prompt(self, kwargs: dict, remediation_prompt: str) -> dict:
        """Inject prompt for generic operations."""
        enhanced_kwargs = kwargs.copy()
        
        # Try common parameter names
        prompt_params = ['input', 'prompt', 'text', 'query', 'message']
        for param in prompt_params:
            if param in enhanced_kwargs and isinstance(enhanced_kwargs[param], str):
                original_value = enhanced_kwargs[param]
                enhanced_kwargs[param] = f"{remediation_prompt}\n\n{original_value}"
                logging.info(f"💉 GENERIC INJECTION: Enhanced {param} parameter")
                break
        
        return enhanced_kwargs

    def _apply_llm_prompt_injection(self, args: tuple, kwargs: dict, class_name: str, 
                                  method_name: str, instance: Any) -> tuple:
        """Apply prompt injection to LLM calls by modifying the messages/prompt."""
        enhanced_args = list(args)
        enhanced_kwargs = kwargs.copy()
        
        # Check if we have any pending remediation prompts to inject
        if not hasattr(self.error_detector, 'pending_remediation_prompts'):
            return tuple(enhanced_args), enhanced_kwargs
        
        pending_prompts = getattr(self.error_detector, 'pending_remediation_prompts', [])
        if not pending_prompts:
            return tuple(enhanced_args), enhanced_kwargs
        
        # Get the most recent remediation prompt
        remediation_prompt = pending_prompts[-1]
        
        logging.info(f"💉 LLM PROMPT INJECTION: Applying remediation prompt to {class_name}.{method_name}")
        logging.info(f"💉 REMEDIATION PROMPT: {remediation_prompt[:200]}...")
        
        # CRITICAL: Never modify LangChain-specific parameters that could break the agent
        langchain_protected_params = {
            'agent_scratchpad', 'intermediate_steps', 'messages', 'input_variables',
            'stop', 'stop_sequences', 'callbacks', 'tags', 'metadata', 'config',
            'run_name', 'run_id', 'parent_run_id', 'run_type'
        }
        
        # Only apply prompt injection to direct LLM calls, not agent calls
        if 'Agent' in class_name or 'Chain' in class_name:
            logging.info(f"🚫 SKIPPING PROMPT INJECTION: {class_name} is an agent/chain, not a direct LLM call")
            return tuple(enhanced_args), enhanced_kwargs
        
        # Handle different LLM input formats - be very careful about what we modify
        if args and len(args) > 0:
            # First argument is typically the input
            input_arg = args[0]
            
            if isinstance(input_arg, str):
                # Simple string input - prepend remediation context
                enhanced_args[0] = f"{remediation_prompt}\n\n{input_arg}"
                logging.info(f"💉 INJECTED: Enhanced string input with remediation context")
                
            elif isinstance(input_arg, list) and input_arg and isinstance(input_arg[0], dict):
                # List of message dictionaries (ChatOpenAI format)
                messages = input_arg.copy()
                
                # Add system message with remediation context
                system_message = {
                    "role": "system",
                    "content": f"{remediation_prompt}\n\nYou are an AI assistant. Please follow the guidance above when responding."
                }
                
                # Insert system message at the beginning
                messages.insert(0, system_message)
                enhanced_args[0] = messages
                logging.info(f"💉 INJECTED: Added system message with remediation context to {len(messages)} messages")
                
            elif isinstance(input_arg, dict):
                # Dictionary input - be very careful about what we modify
                if 'messages' in input_arg and 'agent_scratchpad' not in input_arg:
                    # Only modify if it's not an agent call
                    messages = input_arg['messages'].copy() if isinstance(input_arg['messages'], list) else [input_arg['messages']]
                    
                    system_message = {
                        "role": "system", 
                        "content": f"{remediation_prompt}\n\nYou are an AI assistant. Please follow the guidance above when responding."
                    }
                    messages.insert(0, system_message)
                    
                    enhanced_args[0] = {**input_arg, 'messages': messages}
                    logging.info(f"💉 INJECTED: Enhanced dict input with system message")
                elif 'input' in input_arg and 'agent_scratchpad' not in input_arg:
                    # Only modify if it's not an agent call
                    enhanced_args[0] = {
                        **input_arg,
                        'input': f"{remediation_prompt}\n\n{input_arg['input']}"
                    }
                    logging.info(f"💉 INJECTED: Enhanced dict input field")
                else:
                    logging.info(f"🚫 SKIPPING: Dict input contains protected agent parameters")
        
        # Also check kwargs for input parameters - but be very selective
        safe_prompt_keys = ['input', 'prompt', 'text', 'query']
        for key in safe_prompt_keys:
            if (key in enhanced_kwargs and 
                key not in langchain_protected_params and
                'agent_scratchpad' not in enhanced_kwargs):  # Extra safety check
                
                value = enhanced_kwargs[key]
                
                if isinstance(value, str):
                    enhanced_kwargs[key] = f"{remediation_prompt}\n\n{value}"
                    logging.info(f"💉 INJECTED: Enhanced kwargs['{key}'] with remediation context")
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # Handle message list in kwargs
                    messages = value.copy()
                    system_message = {
                        "role": "system",
                        "content": f"{remediation_prompt}\n\nYou are an AI assistant. Please follow the guidance above when responding."
                    }
                    messages.insert(0, system_message)
                    enhanced_kwargs[key] = messages
                    logging.info(f"💉 INJECTED: Enhanced kwargs['{key}'] with system message")
            else:
                if key in enhanced_kwargs:
                    logging.info(f"🚫 SKIPPING: {key} is protected or agent context detected")
        
        # Clear the used remediation prompt
        if hasattr(self.error_detector, 'pending_remediation_prompts'):
            self.error_detector.pending_remediation_prompts.clear()
        
        return tuple(enhanced_args), enhanced_kwargs


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