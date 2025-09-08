"""
LangChain interceptor for real-time error detection and monitoring.
"""

import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Union, List
from datetime import datetime

from ..core.error_handling.error_detector import ErrorDetector
from ..core.types.error_types import ErrorContext
from ..reporting.logger import AigieLogger

# LangChain imports
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.callbacks.manager import CallbackManagerForChainRun, CallbackManagerForToolRun
    from langchain_core.tracers.context import tracing_v2_enabled
    from langchain_core.outputs import LLMResult, ChatResult
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for when LangChain is not available
    BaseCallbackHandler = object
    CallbackManagerForChainRun = None
    CallbackManagerForToolRun = None
    tracing_v2_enabled = lambda: False
    LLMResult = None
    ChatResult = None
    BaseMessage = None
    LANGCHAIN_AVAILABLE = False


class LangChainInterceptor:
    """Intercepts LangChain operations to detect errors and monitor performance."""
    
    def __init__(self, error_detector: ErrorDetector, logger: AigieLogger):
        self.error_detector = error_detector
        self.logger = logger
        self.intercepted_classes = set()
        self.original_methods = {}
        
        # Initialize the callback handler
        self.callback_handler = LangChainCallbackHandler(error_detector, logger)
        
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
        
        # Register callback handler with LangChain components
        self._register_callback_handler()
    
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
    
    def _register_callback_handler(self):
        """Register the callback handler with LangChain components."""
        if not LANGCHAIN_AVAILABLE:
            self.logger.log_system_event("LangChain not available - skipping callback registration")
            return
        
        try:
            # Register with global callback manager if available
            from langchain_core.callbacks import get_callback_manager
            callback_manager = get_callback_manager()
            if callback_manager:
                callback_manager.add_handler(self.callback_handler)
                self.logger.log_system_event("Callback handler registered with global callback manager")
            
            # Also register with tracing if available
            if tracing_v2_enabled():
                from langchain_core.tracers import LangChainTracer
                tracer = LangChainTracer()
                tracer.add_handler(self.callback_handler)
                self.logger.log_system_event("Callback handler registered with LangChain tracer")
                
        except Exception as e:
            self.logger.log_system_event(f"Could not register callback handler: {e}")
    
    def get_callback_handler(self):
        """Get the callback handler for manual registration."""
        return self.callback_handler
    
    def register_with_langchain_component(self, component):
        """Register the callback handler with a specific LangChain component."""
        if not LANGCHAIN_AVAILABLE:
            self.logger.log_system_event("LangChain not available - cannot register callback handler")
            return False
        
        try:
            # Try to add the callback handler to the component
            if hasattr(component, 'callbacks'):
                if component.callbacks is None:
                    component.callbacks = [self.callback_handler]
                else:
                    if isinstance(component.callbacks, list):
                        component.callbacks.append(self.callback_handler)
                    else:
                        component.callbacks = [component.callbacks, self.callback_handler]
                self.logger.log_system_event(f"Callback handler registered with {type(component).__name__}")
                return True
            elif hasattr(component, 'callback_manager'):
                component.callback_manager.add_handler(self.callback_handler)
                self.logger.log_system_event(f"Callback handler registered with {type(component).__name__} callback manager")
                return True
            else:
                self.logger.log_system_event(f"Component {type(component).__name__} does not support callbacks")
                return False
        except Exception as e:
            self.logger.log_system_event(f"Failed to register callback handler with {type(component).__name__}: {e}")
            return False
    
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
                enhanced_args, enhanced_kwargs = self._apply_unified_prompt_injection(
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
                enhanced_args, enhanced_kwargs = self._apply_unified_prompt_injection(
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
    
    def _apply_unified_prompt_injection(self, args: tuple, kwargs: dict, class_name: str, 
                                      method_name: str, instance: Any) -> tuple:
        """Apply intelligent prompt injection based on operation type and context."""
        enhanced_args = list(args)
        enhanced_kwargs = kwargs.copy()
        
        # Check if we have pending remediation prompts
        if (hasattr(self.error_detector, 'pending_remediation_prompts') and 
            self.error_detector.pending_remediation_prompts):
            
            remediation_prompt = self.error_detector.pending_remediation_prompts[0]
            logging.info(f"💉 SMART INJECTION: Applying remediation prompt for {class_name}.{method_name}")
            
            # Determine operation type and apply appropriate injection strategy
            operation_type = self._determine_operation_type(class_name, method_name)
            enhanced_kwargs = self._inject_prompt_by_type(enhanced_kwargs, remediation_prompt, operation_type)
            
            # Clear the used remediation prompt
            self.error_detector.pending_remediation_prompts.clear()
        
        return tuple(enhanced_args), enhanced_kwargs
    
    def _determine_operation_type(self, class_name: str, method_name: str) -> str:
        """Determine the operation type for prompt injection strategy."""
        # Agent operations
        agent_classes = ['AgentExecutor', 'Agent', 'ConversationalAgent', 'ReActAgent']
        if any(agent_class in class_name for agent_class in agent_classes):
            return 'agent'
        
        # LLM operations
        llm_classes = ['OpenAI', 'ChatOpenAI', 'Anthropic', 'ChatAnthropic', 'LLM', 'ChatModel']
        if any(llm_class in class_name for llm_class in llm_classes):
            return 'llm'
        
        # Chain operations
        chain_classes = ['Chain', 'LLMChain', 'ConversationChain', 'RetrievalQA']
        if any(chain_class in class_name for chain_class in chain_classes):
            return 'chain'
        
        # Tool operations
        tool_classes = ['BaseTool', 'StructuredTool', 'Tool']
        if any(tool_class in class_name for tool_class in tool_classes):
            return 'tool'
        
        # Default to generic
        return 'generic'
    
    def _inject_prompt_by_type(self, kwargs: dict, remediation_prompt: str, operation_type: str) -> dict:
        """Inject prompt based on operation type with appropriate strategy."""
        enhanced_kwargs = kwargs.copy()
        
        if operation_type == 'agent':
            # For agent operations, inject via input parameter
            if 'input' in enhanced_kwargs:
                original_input = enhanced_kwargs['input']
                enhanced_kwargs['input'] = f"{remediation_prompt}\n\n{original_input}"
                logging.info("💉 AGENT INJECTION: Enhanced input parameter")
            elif 'inputs' in enhanced_kwargs and isinstance(enhanced_kwargs['inputs'], dict):
                if 'input' in enhanced_kwargs['inputs']:
                    original_input = enhanced_kwargs['inputs']['input']
                    enhanced_kwargs['inputs']['input'] = f"{remediation_prompt}\n\n{original_input}"
                    logging.info("💉 AGENT INJECTION: Enhanced inputs['input'] parameter")
        
        elif operation_type == 'llm':
            # For LLM operations, inject via messages or prompt parameter
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
        
        elif operation_type == 'chain':
            # For chain operations, inject via input parameter
            if 'input' in enhanced_kwargs:
                original_input = enhanced_kwargs['input']
                enhanced_kwargs['input'] = f"{remediation_prompt}\n\n{original_input}"
                logging.info("💉 CHAIN INJECTION: Enhanced input parameter")
            elif 'inputs' in enhanced_kwargs and isinstance(enhanced_kwargs['inputs'], dict):
                if 'input' in enhanced_kwargs['inputs']:
                    original_input = enhanced_kwargs['inputs']['input']
                    enhanced_kwargs['inputs']['input'] = f"{remediation_prompt}\n\n{original_input}"
                    logging.info("💉 CHAIN INJECTION: Enhanced inputs['input'] parameter")
        
        else:
            # Generic injection for other operation types
            prompt_params = ['input', 'prompt', 'text', 'query', 'message']
            for param in prompt_params:
                if param in enhanced_kwargs and isinstance(enhanced_kwargs[param], str):
                    original_value = enhanced_kwargs[param]
                    enhanced_kwargs[param] = f"{remediation_prompt}\n\n{original_value}"
                    logging.info(f"💉 GENERIC INJECTION: Enhanced {param} parameter")
                    break
        
        return enhanced_kwargs
    



class LangChainCallbackHandler(BaseCallbackHandler):
    """Enhanced LangChain callback handler with proper BaseCallbackHandler inheritance."""
    
    def __init__(self, error_detector: ErrorDetector, logger: AigieLogger):
        if LANGCHAIN_AVAILABLE:
            super().__init__()
        self.error_detector = error_detector
        self.logger = logger
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when a chain starts."""
        if not self.error_detector.is_monitoring:
            return
            
        # Log chain start
        self.logger.log_system_event(
            "Chain started",
            {
                "chain_name": serialized.get("name", "unknown"),
                "chain_id": serialized.get("id", "unknown"),
                "inputs": str(inputs)[:200],
                "run_id": kwargs.get("run_id"),
                "parent_run_id": kwargs.get("parent_run_id")
            }
        )
        
        # Create enhanced error context for monitoring
        context = ErrorContext(
            timestamp=datetime.now(),
            framework="langchain",
            component="Chain",
            method="run",
            input_data=inputs,
            run_id=kwargs.get("run_id"),
            parent_run_id=kwargs.get("parent_run_id"),
            serialized=serialized,
            inputs=inputs,
            tags=kwargs.get("tags", []),
            metadata=kwargs.get("metadata", {})
        )
        
        # Store context for potential error handling
        self.error_detector.store_operation_for_retry(
            f"chain_{kwargs.get('run_id', 'unknown')}", 
            None, inputs, {}, context
        )
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when a chain ends."""
        if not self.error_detector.is_monitoring:
            return
            
        self.logger.log_system_event(
            "Chain completed",
            {
                "outputs": str(outputs)[:200],
                "run_id": kwargs.get("run_id"),
                "parent_run_id": kwargs.get("parent_run_id")
            }
        )
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs):
        """Called when a chain encounters an error."""
        if not self.error_detector.is_monitoring:
            return
            
        # Create error context
        context = self._create_langchain_error_context(error, "Chain", "run", **kwargs)
        
        # Handle with chain-specific remediation
        return self._handle_callback_error(error, context, "Chain", **kwargs)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when a tool starts."""
        if not self.error_detector.is_monitoring:
            return
            
        self.logger.log_system_event(
            "Tool started",
            {
                "tool_name": serialized.get("name", "unknown"),
                "tool_id": serialized.get("id", "unknown"),
                "input": input_str[:200],
                "run_id": kwargs.get("run_id"),
                "parent_run_id": kwargs.get("parent_run_id")
            }
        )
        
        # Create enhanced error context for monitoring
        context = ErrorContext(
            timestamp=datetime.now(),
            framework="langchain",
            component="Tool",
            method="run",
            input_data={"input_str": input_str},
            run_id=kwargs.get("run_id"),
            parent_run_id=kwargs.get("parent_run_id"),
            serialized=serialized,
            tool_metadata=serialized,
            tags=kwargs.get("tags", []),
            metadata=kwargs.get("metadata", {})
        )
        
        # Store context for potential error handling
        self.error_detector.store_operation_for_retry(
            f"tool_{kwargs.get('run_id', 'unknown')}", 
            None, (input_str,), {}, context
        )
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when a tool ends."""
        if not self.error_detector.is_monitoring:
            return
            
        self.logger.log_system_event(
            "Tool completed",
            {
                "output": output[:200],
                "run_id": kwargs.get("run_id"),
                "parent_run_id": kwargs.get("parent_run_id")
            }
        )
    
    def _create_langchain_error_context(self, error: Exception, component: str, method: str, **kwargs) -> ErrorContext:
        """Create a standardized error context for LangChain callbacks."""
        return ErrorContext(
            timestamp=datetime.now(),
            framework="langchain",
            component=component,
            method=method,
            stack_trace=str(error),
            langchain_error_type=type(error).__name__,
            run_id=kwargs.get("run_id"),
            parent_run_id=kwargs.get("parent_run_id"),
            error=error,
            serialized=kwargs.get("serialized", {}),
            tool_metadata=kwargs.get("serialized", {}) if component == "Tool" else None,
            tags=kwargs.get("tags", []),
            metadata=kwargs.get("metadata", {})
        )
    
    def _handle_callback_error(self, error: Exception, context: ErrorContext, error_type: str, **kwargs):
        """Handle error processing for LangChain callbacks with type-specific remediation."""
        # Use the error detector's enhanced analysis
        detected_error = self.error_detector._detect_error(error, context)
        
        # Attempt automatic remediation if enabled
        if self.error_detector.enable_automatic_retry:
            remediation_result = self.error_detector._attempt_automatic_retry(
                error, context, detected_error
            )
            if remediation_result:
                self.logger.log_system_event(
                    f"{error_type} error remediated automatically",
                    {"run_id": kwargs.get("run_id"), "remediation_result": str(remediation_result)[:200]}
                )
                return remediation_result
        
        # Log the error
        self.logger.log_error(f"{error_type} error: {error}", context)
        return None

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs):
        """Called when a tool encounters an error."""
        if not self.error_detector.is_monitoring:
            return
            
        # Create error context
        context = self._create_langchain_error_context(error, "Tool", "run", **kwargs)
        
        # Handle with tool-specific remediation
        return self._handle_callback_error(error, context, "Tool", **kwargs)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when an LLM starts."""
        if not self.error_detector.is_monitoring:
            return
            
        self.logger.log_system_event(
            "LLM started",
            {
                "llm_name": serialized.get("name", "unknown"),
                "llm_id": serialized.get("id", "unknown"),
                "prompts_count": len(prompts),
                "run_id": kwargs.get("run_id"),
                "parent_run_id": kwargs.get("parent_run_id")
            }
        )
    
    def on_llm_end(self, response: Union[LLMResult, ChatResult], **kwargs):
        """Called when an LLM ends."""
        if not self.error_detector.is_monitoring:
            return
            
        self.logger.log_system_event(
            "LLM completed",
            {
                "generations_count": len(response.generations) if hasattr(response, 'generations') else 0,
                "run_id": kwargs.get("run_id"),
                "parent_run_id": kwargs.get("parent_run_id")
            }
        )
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs):
        """Called when an LLM encounters an error."""
        if not self.error_detector.is_monitoring:
            return
            
        # Create error context
        context = self._create_langchain_error_context(error, "LLM", "generate", **kwargs)
        
        # Handle with LLM-specific remediation
        return self._handle_callback_error(error, context, "LLM", **kwargs)
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs):
        """Called when a retriever starts."""
        if not self.error_detector.is_monitoring:
            return
            
        self.logger.log_system_event(
            "Retriever started",
            {
                "retriever_name": serialized.get("name", "unknown"),
                "query": query[:200],
                "run_id": kwargs.get("run_id"),
                "parent_run_id": kwargs.get("parent_run_id")
            }
        )
    
    def on_retriever_end(self, documents: List[Any], **kwargs):
        """Called when a retriever ends."""
        if not self.error_detector.is_monitoring:
            return
            
        self.logger.log_system_event(
            "Retriever completed",
            {
                "documents_count": len(documents),
                "run_id": kwargs.get("run_id"),
                "parent_run_id": kwargs.get("parent_run_id")
            }
        )
    
    def on_retriever_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs):
        """Called when a retriever encounters an error."""
        if not self.error_detector.is_monitoring:
            return
            
        # Create error context
        context = self._create_langchain_error_context(error, "Retriever", "get_relevant_documents", **kwargs)
        
        # Handle with retriever-specific remediation
        return self._handle_callback_error(error, context, "Retriever", **kwargs)