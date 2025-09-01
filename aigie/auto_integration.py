"""
Automatic integration system for Aigie.
"""

import os
import sys
from typing import Optional, Dict, Any
from contextlib import contextmanager

from .core.error_detector import ErrorDetector
from .reporting.logger import AigieLogger, ConsoleLogger
from .interceptors.langchain import LangChainInterceptor
from .interceptors.langgraph import LangGraphInterceptor
from .utils.config import AigieConfig


class AigieAutoIntegrator:
    """Automatically integrates Aigie monitoring into LangChain and LangGraph applications."""
    
    def __init__(self, config: Optional[AigieConfig] = None):
        self.config = config or AigieConfig()
        self.error_detector = ErrorDetector(
            enable_performance_monitoring=self.config.enable_performance_monitoring,
            enable_resource_monitoring=self.config.enable_resource_monitoring,
            enable_gemini_analysis=self.config.enable_gemini_analysis,
            gemini_project_id=self.config.gemini_project_id,
            gemini_location=self.config.gemini_location
        )
        
        # Initialize logger based on environment
        self.logger = self._setup_logger()
        
        # Initialize interceptors
        self.langchain_interceptor = LangChainInterceptor(self.error_detector, self.logger)
        self.langgraph_interceptor = LangGraphInterceptor(self.error_detector, self.logger)
        
        # Integration status
        self.is_integrated = False
        self.integration_start_time = None
        
        # Add error handler to log errors
        self.error_detector.add_error_handler(self.logger.log_error)
    
    def _setup_logger(self) -> AigieLogger:
        """Setup logger based on configuration and environment."""
        if self.config.enable_rich_formatting:
            return ConsoleLogger(log_level=self.config.log_level)
        else:
            return AigieLogger(
                log_level=self.config.log_level,
                enable_console=self.config.enable_console,
                enable_file=self.config.enable_file,
                log_file_path=self.config.log_file_path,
                enable_rich_formatting=self.config.enable_rich_formatting
            )
    
    def auto_integrate(self):
        """Automatically integrate Aigie monitoring."""
        if self.is_integrated:
            self.logger.log_system_event("Aigie already integrated")
            return
        
        try:
            # Start error detection
            self.error_detector.start_monitoring()
            
            # Start LangChain interception
            self.langchain_interceptor.start_intercepting()
            
            # Start LangGraph interception
            self.langgraph_interceptor.start_intercepting()
            
            self.is_integrated = True
            self.integration_start_time = self.error_detector.performance_monitor.start_monitoring(
                "AigieAutoIntegrator", "auto_integrate"
            )
            
            self.logger.log_system_event("🎯 Aigie auto-integration completed successfully!")
            self.logger.log_system_event("🚀 Monitoring LangChain and LangGraph operations in real-time")
            
            # Display initial status
            self._display_integration_status()
            
            # Display Gemini status if enabled
            if self.config.enable_gemini_analysis:
                self._display_gemini_status()
                
        except Exception as e:
            self.logger.log_system_event(f"Failed to auto-integrate Aigie: {e}")
            raise
    
    def stop_integration(self):
        """Stop Aigie monitoring and restore original methods."""
        if not self.is_integrated:
            return
        
        try:
            # Stop interceptors
            self.langchain_interceptor.stop_intercepting()
            self.langgraph_interceptor.stop_intercepting()
            
            # Stop error detection
            self.error_detector.stop_monitoring()
            
            self.is_integrated = False
            self.logger.log_system_event("🛑 Aigie integration stopped")
            
        except Exception as e:
            self.logger.log_system_event(f"Error stopping Aigie integration: {e}")
    
    def _display_integration_status(self):
        """Display current integration status."""
        if not hasattr(self.logger, 'console'):
            return
        
        # Get status information
        lc_status = self.langchain_interceptor.get_interception_status()
        lg_status = self.langgraph_interceptor.get_interception_status()
        
        # Display status
        self.logger.console.print("📊 Integration Status:", style="bold blue")
        self.logger.console.print(f"  LangChain: {'✅ Active' if lc_status['is_intercepting'] else '❌ Inactive'}")
        self.logger.console.print(f"  LangGraph: {'✅ Active' if lg_status['is_intercepting'] else '❌ Inactive'}")
        self.logger.console.print(f"  Error Detection: {'✅ Active' if self.error_detector.is_monitoring else '❌ Inactive'}")
        
        if lc_status['intercepted_classes']:
            self.logger.console.print(f"  LangChain Classes: {', '.join(lc_status['intercepted_classes'])}")
        
        if lg_status['intercepted_classes']:
            self.logger.console.print(f"  LangGraph Classes: {', '.join(lg_status['intercepted_classes'])}")
    
    def _display_gemini_status(self):
        """Display Gemini integration status."""
        if not hasattr(self.logger, 'console'):
            return
        
        gemini_status = self.error_detector.get_gemini_status()
        
        if gemini_status.get('enabled', False):
            self.logger.console.print("🤖 Gemini Status:", style="bold green")
            self.logger.console.print(f"  Project: {gemini_status.get('project_id', 'N/A')}")
            self.logger.console.print(f"  Location: {gemini_status.get('location', 'N/A')}")
            self.logger.console.print(f"  Model: {'✅ Loaded' if gemini_status.get('model_loaded') else '❌ Not Loaded'}")
        else:
            self.logger.console.print("🤖 Gemini Status:", style="bold yellow")
            self.logger.console.print(f"  Status: {'❌ Disabled' if not gemini_status.get('enabled') else '⚠️  Not Available'}")
            if gemini_status.get('reason'):
                self.logger.console.print(f"  Reason: {gemini_status['reason']}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        return {
            "is_integrated": self.is_integrated,
            "integration_start_time": self.integration_start_time.isoformat() if self.integration_start_time else None,
            "error_detector": {
                "is_monitoring": self.error_detector.is_monitoring,
                "total_errors": len(self.error_detector.error_history),
                "recent_errors": self.error_detector.get_error_summary(window_minutes=5)
            },
            "langchain": self.langchain_interceptor.get_interception_status(),
            "langgraph": self.langgraph_interceptor.get_interception_status(),
            "system_health": self.error_detector.get_system_health(),
            "configuration": self.config.to_dict()
        }
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of monitored operations."""
        analysis = {
            "error_summary": self.error_detector.get_error_summary(),
            "performance_summary": self.error_detector.performance_monitor.get_performance_summary() if self.error_detector.performance_monitor else None,
            "langchain_analysis": {
                "interception_status": self.langchain_interceptor.get_interception_status()
            },
            "langgraph_analysis": {
                "interception_status": self.langgraph_interceptor.get_interception_status(),
                "graph_analysis": self.langgraph_interceptor.get_graph_analysis(),
                "node_execution_stats": self.langgraph_interceptor.get_node_execution_stats(),
                "state_transition_analysis": self.langgraph_interceptor.get_state_transition_analysis()
            }
        }
        
        return analysis
    
    @contextmanager
    def temporary_integration(self):
        """Context manager for temporary integration."""
        try:
            self.auto_integrate()
            yield self
        finally:
            self.stop_integration()


# Global instance for easy access
_aigie_integrator: Optional[AigieAutoIntegrator] = None


def auto_integrate(config: Optional[AigieConfig] = None) -> AigieAutoIntegrator:
    """Automatically integrate Aigie monitoring."""
    global _aigie_integrator
    
    if _aigie_integrator is None:
        _aigie_integrator = AigieAutoIntegrator(config)
    
    _aigie_integrator.auto_integrate()
    return _aigie_integrator


def get_integrator() -> Optional[AigieAutoIntegrator]:
    """Get the current Aigie integrator instance."""
    return _aigie_integrator


def stop_integration():
    """Stop Aigie integration."""
    global _aigie_integrator
    
    if _aigie_integrator:
        _aigie_integrator.stop_integration()
        _aigie_integrator = None


def get_status() -> Optional[Dict[str, Any]]:
    """Get current integration status."""
    if _aigie_integrator:
        return _aigie_integrator.get_integration_status()
    return None


def get_analysis() -> Optional[Dict[str, Any]]:
    """Get detailed analysis."""
    if _aigie_integrator:
        return _aigie_integrator.get_detailed_analysis()
    return None


# Environment-based auto-integration
def _check_auto_integration_env():
    """Check if auto-integration should be enabled via environment variables."""
    auto_enable = os.getenv("AIGIE_AUTO_ENABLE", "false").lower() == "true"
    
    if auto_enable:
        config = AigieConfig.from_environment()
        auto_integrate(config)


# Auto-integration on import if enabled
if os.getenv("AIGIE_AUTO_ENABLE", "false").lower() == "true":
    _check_auto_integration_env()


# Convenience functions for manual integration
def enable_monitoring(config: Optional[AigieConfig] = None):
    """Enable Aigie monitoring manually."""
    return auto_integrate(config)


def disable_monitoring():
    """Disable Aigie monitoring."""
    stop_integration()


def show_status():
    """Display current monitoring status."""
    integrator = get_integrator()
    if integrator and hasattr(integrator.logger, 'console'):
        integrator._display_integration_status()
        
        # Show error summary if any
        if integrator.error_detector.error_history:
            integrator.logger.display_error_summary()
        
        # Show system health
        health_data = integrator.error_detector.get_system_health()
        integrator.logger.display_system_health(health_data)


def show_analysis():
    """Display detailed analysis."""
    integrator = get_integrator()
    if integrator and hasattr(integrator.logger, 'console'):
        analysis = integrator.get_detailed_analysis()
        
        # Display analysis in console
        integrator.logger.console.print("📊 Detailed Analysis", style="bold blue")
        
        # Error summary
        if analysis.get("error_summary"):
            error_summary = analysis["error_summary"]
            integrator.logger.console.print(f"🚨 Total Errors: {error_summary.get('total_errors', 0)}")
            
            if error_summary.get("severity_distribution"):
                for severity, count in error_summary["severity_distribution"].items():
                    integrator.logger.console.print(f"  {severity.upper()}: {count}")
        
        # Performance summary
        if analysis.get("performance_summary"):
            perf = analysis["performance_summary"]
            integrator.logger.console.print(f"📈 Total Executions: {perf.get('total_executions', 0)}")
            integrator.logger.console.print(f"⏱️  Avg Execution Time: {perf.get('avg_execution_time', 0):.2f}s")
        
        # LangGraph analysis
        if analysis.get("langgraph_analysis"):
            lg_analysis = analysis["langgraph_analysis"]
            if lg_analysis.get("graph_analysis"):
                graph_count = lg_analysis["graph_analysis"]["total_graphs"]
                integrator.logger.console.print(f"🕸️  Tracked Graphs: {graph_count}")
            
            if lg_analysis.get("node_execution_stats"):
                node_count = lg_analysis["node_execution_stats"]["total_nodes"]
                integrator.logger.console.print(f"🔗 Tracked Nodes: {node_count}")


# Magic methods for IPython/Jupyter integration
def _ipython_display_():
    """IPython display method."""
    show_status()


# Register with IPython if available
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        # Add magic commands
        ipython.register_magic_function(show_status, 'line', 'aigie_status')
        ipython.register_magic_function(show_analysis, 'line', 'aigie_analysis')
        ipython.register_magic_function(enable_monitoring, 'line', 'aigie_enable')
        ipython.register_magic_function(disable_monitoring, 'line', 'aigie_disable')
except ImportError:
    pass
