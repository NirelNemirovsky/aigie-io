"""
Configuration management for Aigie.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class AigieConfig:
    """Configuration for Aigie monitoring and error detection."""
    
    # Logging configuration
    log_level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = False
    log_file_path: Optional[str] = None
    enable_rich_formatting: bool = True
    
    # Error detection configuration
    enable_performance_monitoring: bool = True
    enable_resource_monitoring: bool = True
    enable_timeout_detection: bool = True
    timeout_threshold: float = 300.0  # 5 minutes
    enable_memory_leak_detection: bool = True
    memory_leak_threshold: float = 100.0  # MB
    
    # Performance monitoring thresholds
    execution_time_threshold: float = 30.0  # seconds
    memory_threshold: float = 1024.0  # MB
    cpu_threshold: float = 80.0  # percent
    
    # System resource thresholds
    memory_usage_threshold: float = 0.9  # 90%
    cpu_usage_threshold: float = 0.8  # 80%
    disk_usage_threshold: float = 0.9  # 90%
    
    # Auto-integration settings
    auto_enable: bool = False
    auto_cleanup: bool = True
    
    # Cloud integration
    enable_cloud_logging: bool = False
    cloud_project_id: Optional[str] = None
    cloud_location: Optional[str] = None
    
    # Gemini integration settings
    enable_gemini_analysis: bool = True
    gemini_project_id: Optional[str] = None
    gemini_location: str = "us-central1"
    gemini_api_key: Optional[str] = None
    
    # Retry settings
    enable_automatic_retry: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_confidence_threshold: float = 0.7
    
    # Advanced settings
    max_error_history: int = 1000
    max_performance_history: int = 1000
    cleanup_interval_minutes: int = 60
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.log_level}. Must be one of {valid_log_levels}")
        
        # Validate thresholds
        if self.timeout_threshold <= 0:
            raise ValueError("Timeout threshold must be positive")
        
        if self.memory_leak_threshold <= 0:
            raise ValueError("Memory leak threshold must be positive")
        
        if self.execution_time_threshold <= 0:
            raise ValueError("Execution time threshold must be positive")
        
        if self.memory_threshold <= 0:
            raise ValueError("Memory threshold must be positive")
        
        if self.cpu_threshold <= 0 or self.cpu_threshold > 100:
            raise ValueError("CPU threshold must be between 0 and 100")
        
        if self.memory_usage_threshold <= 0 or self.memory_usage_threshold > 1:
            raise ValueError("Memory usage threshold must be between 0 and 1")
        
        if self.cpu_usage_threshold <= 0 or self.cpu_usage_threshold > 1:
            raise ValueError("CPU usage threshold must be between 0 and 1")
        
        if self.disk_usage_threshold <= 0 or self.disk_usage_threshold > 1:
            raise ValueError("Disk usage threshold must be between 0 and 1")
        
        if self.max_error_history <= 0:
            raise ValueError("Max error history must be positive")
        
        if self.max_performance_history <= 0:
            raise ValueError("Max performance history must be positive")
        
        if self.cleanup_interval_minutes <= 0:
            raise ValueError("Cleanup interval must be positive")
    
    @classmethod
    def from_environment(cls) -> 'AigieConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Logging configuration
        if os.getenv("AIGIE_LOG_LEVEL"):
            config.log_level = os.getenv("AIGIE_LOG_LEVEL", "INFO")
        
        if os.getenv("AIGIE_ENABLE_CONSOLE"):
            config.enable_console = os.getenv("AIGIE_ENABLE_CONSOLE", "true").lower() == "true"
        
        if os.getenv("AIGIE_ENABLE_FILE"):
            config.enable_file = os.getenv("AIGIE_ENABLE_FILE", "false").lower() == "true"
        
        if os.getenv("AIGIE_LOG_FILE_PATH"):
            config.log_file_path = os.getenv("AIGIE_LOG_FILE_PATH")
        
        if os.getenv("AIGIE_ENABLE_RICH_FORMATTING"):
            config.enable_rich_formatting = os.getenv("AIGIE_ENABLE_RICH_FORMATTING", "true").lower() == "true"
        
        # Error detection configuration
        if os.getenv("AIGIE_ENABLE_PERFORMANCE_MONITORING"):
            config.enable_performance_monitoring = os.getenv("AIGIE_ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
        
        if os.getenv("AIGIE_ENABLE_RESOURCE_MONITORING"):
            config.enable_resource_monitoring = os.getenv("AIGIE_ENABLE_RESOURCE_MONITORING", "true").lower() == "true"
        
        if os.getenv("AIGIE_ENABLE_TIMEOUT_DETECTION"):
            config.enable_timeout_detection = os.getenv("AIGIE_ENABLE_TIMEOUT_DETECTION", "true").lower() == "true"
        
        if os.getenv("AIGIE_TIMEOUT_THRESHOLD"):
            try:
                config.timeout_threshold = float(os.getenv("AIGIE_TIMEOUT_THRESHOLD", "300.0"))
            except ValueError:
                pass
        
        if os.getenv("AIGIE_ENABLE_MEMORY_LEAK_DETECTION"):
            config.enable_memory_leak_detection = os.getenv("AIGIE_ENABLE_MEMORY_LEAK_DETECTION", "true").lower() == "true"
        
        if os.getenv("AIGIE_MEMORY_LEAK_THRESHOLD"):
            try:
                config.memory_leak_threshold = float(os.getenv("AIGIE_MEMORY_LEAK_THRESHOLD", "100.0"))
            except ValueError:
                pass
        
        # Performance monitoring thresholds
        if os.getenv("AIGIE_EXECUTION_TIME_THRESHOLD"):
            try:
                config.execution_time_threshold = float(os.getenv("AIGIE_EXECUTION_TIME_THRESHOLD", "30.0"))
            except ValueError:
                pass
        
        if os.getenv("AIGIE_MEMORY_THRESHOLD"):
            try:
                config.memory_threshold = float(os.getenv("AIGIE_MEMORY_THRESHOLD", "1024.0"))
            except ValueError:
                pass
        
        if os.getenv("AIGIE_CPU_THRESHOLD"):
            try:
                config.cpu_threshold = float(os.getenv("AIGIE_CPU_THRESHOLD", "80.0"))
            except ValueError:
                pass
        
        # System resource thresholds
        if os.getenv("AIGIE_MEMORY_USAGE_THRESHOLD"):
            try:
                config.memory_usage_threshold = float(os.getenv("AIGIE_MEMORY_USAGE_THRESHOLD", "0.9"))
            except ValueError:
                pass
        
        if os.getenv("AIGIE_CPU_USAGE_THRESHOLD"):
            try:
                config.cpu_usage_threshold = float(os.getenv("AIGIE_CPU_USAGE_THRESHOLD", "0.8"))
            except ValueError:
                pass
        
        if os.getenv("AIGIE_DISK_USAGE_THRESHOLD"):
            try:
                config.disk_usage_threshold = float(os.getenv("AIGIE_DISK_USAGE_THRESHOLD", "0.9"))
            except ValueError:
                pass
        
        # Auto-integration settings
        if os.getenv("AIGIE_AUTO_ENABLE"):
            config.auto_enable = os.getenv("AIGIE_AUTO_ENABLE", "false").lower() == "true"
        
        if os.getenv("AIGIE_AUTO_CLEANUP"):
            config.auto_cleanup = os.getenv("AIGIE_AUTO_CLEANUP", "true").lower() == "true"
        
        # Cloud integration
        if os.getenv("AIGIE_ENABLE_CLOUD_LOGGING"):
            config.enable_cloud_logging = os.getenv("AIGIE_ENABLE_CLOUD_LOGGING", "false").lower() == "true"
        
        if os.getenv("AIGIE_CLOUD_PROJECT_ID"):
            config.cloud_project_id = os.getenv("AIGIE_CLOUD_PROJECT_ID")
        
        if os.getenv("AIGIE_CLOUD_LOCATION"):
            config.cloud_location = os.getenv("AIGIE_CLOUD_LOCATION")
        
        # Gemini integration settings
        if os.getenv("AIGIE_ENABLE_GEMINI"):
            config.enable_gemini_analysis = os.getenv("AIGIE_ENABLE_GEMINI", "true").lower() == "true"
        
        if os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("AIGIE_GEMINI_PROJECT_ID"):
            config.gemini_project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("AIGIE_GEMINI_PROJECT_ID")
        
        if os.getenv("AIGIE_GEMINI_LOCATION"):
            config.gemini_location = os.getenv("AIGIE_GEMINI_LOCATION", "us-central1")
        
        if os.getenv("GEMINI_API_KEY") or os.getenv("AIGIE_GEMINI_API_KEY"):
            config.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("AIGIE_GEMINI_API_KEY")
        
        # Retry settings
        if os.getenv("AIGIE_ENABLE_RETRY"):
            config.enable_automatic_retry = os.getenv("AIGIE_ENABLE_RETRY", "true").lower() == "true"
        
        if os.getenv("AIGIE_MAX_RETRIES"):
            try:
                config.max_retries = int(os.getenv("AIGIE_MAX_RETRIES", "3"))
            except ValueError:
                pass
        
        if os.getenv("AIGIE_RETRY_DELAY"):
            try:
                config.retry_delay = float(os.getenv("AIGIE_RETRY_DELAY", "1.0"))
            except ValueError:
                pass
        
        if os.getenv("AIGIE_RETRY_CONFIDENCE"):
            try:
                config.retry_confidence_threshold = float(os.getenv("AIGIE_RETRY_CONFIDENCE", "0.7"))
            except ValueError:
                pass
        
        # Advanced settings
        if os.getenv("AIGIE_MAX_ERROR_HISTORY"):
            try:
                config.max_error_history = int(os.getenv("AIGIE_MAX_ERROR_HISTORY", "1000"))
            except ValueError:
                pass
        
        if os.getenv("AIGIE_MAX_PERFORMANCE_HISTORY"):
            try:
                config.max_performance_history = int(os.getenv("AIGIE_MAX_PERFORMANCE_HISTORY", "1000"))
            except ValueError:
                pass
        
        if os.getenv("AIGIE_CLEANUP_INTERVAL_MINUTES"):
            try:
                config.cleanup_interval_minutes = int(os.getenv("AIGIE_CLEANUP_INTERVAL_MINUTES", "60"))
            except ValueError:
                pass
        
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> 'AigieConfig':
        """Create configuration from a configuration file."""
        import json
        import yaml
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                config_data = json.load(f)
            elif file_path.endswith(('.yml', '.yaml')):
                config_data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported configuration file format. Use .json, .yml, or .yaml")
        
        # Create config instance
        config = cls()
        
        # Update with file data
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "log_level": self.log_level,
            "enable_console": self.enable_console,
            "enable_file": self.enable_file,
            "log_file_path": self.log_file_path,
            "enable_rich_formatting": self.enable_rich_formatting,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "enable_resource_monitoring": self.enable_resource_monitoring,
            "enable_timeout_detection": self.enable_timeout_detection,
            "timeout_threshold": self.timeout_threshold,
            "enable_memory_leak_detection": self.enable_memory_leak_detection,
            "memory_leak_threshold": self.memory_leak_threshold,
            "execution_time_threshold": self.execution_time_threshold,
            "memory_threshold": self.memory_threshold,
            "cpu_threshold": self.cpu_threshold,
            "memory_usage_threshold": self.memory_usage_threshold,
            "cpu_usage_threshold": self.cpu_usage_threshold,
            "disk_usage_threshold": self.disk_usage_threshold,
            "auto_enable": self.auto_enable,
            "auto_cleanup": self.auto_cleanup,
            "enable_cloud_logging": self.enable_cloud_logging,
            "cloud_project_id": self.cloud_project_id,
            "cloud_location": self.cloud_location,
            "enable_gemini_analysis": self.enable_gemini_analysis,
            "gemini_project_id": self.gemini_project_id,
            "gemini_location": self.gemini_location,
            "enable_automatic_retry": self.enable_automatic_retry,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "retry_confidence_threshold": self.retry_confidence_threshold,
            "max_error_history": self.max_error_history,
            "max_performance_history": self.max_performance_history,
            "cleanup_interval_minutes": self.cleanup_interval_minutes
        }
    
    def save_to_file(self, file_path: str):
        """Save configuration to a file."""
        import json
        import yaml
        
        config_data = self.to_dict()
        
        if file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        elif file_path.endswith(('.yml', '.yaml')):
            with open(file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .json, .yml, or .yaml")
    
    def merge(self, other: 'AigieConfig') -> 'AigieConfig':
        """Merge this configuration with another configuration."""
        merged = AigieConfig()
        
        # Copy current values
        for key, value in self.to_dict().items():
            setattr(merged, key, value)
        
        # Override with other values (only if not None)
        for key, value in other.to_dict().items():
            if value is not None:
                setattr(merged, key, value)
        
        return merged
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get effective configuration with computed values."""
        config = self.to_dict()
        
        # Add computed values
        config["is_auto_enabled"] = self.auto_enable
        config["has_cloud_integration"] = self.enable_cloud_logging and self.cloud_project_id
        
        return config


# Default configurations for different environments
def get_development_config() -> AigieConfig:
    """Get configuration optimized for development."""
    return AigieConfig(
        log_level="DEBUG",
        enable_console=True,
        enable_file=False,
        enable_rich_formatting=True,
        enable_performance_monitoring=True,
        enable_resource_monitoring=True,
        timeout_threshold=60.0,  # Shorter timeout for development
        execution_time_threshold=10.0,  # More sensitive to slow execution
        auto_cleanup=False  # Keep history for debugging
    )


def get_production_config() -> AigieConfig:
    """Get configuration optimized for production."""
    return AigieConfig(
        log_level="INFO",
        enable_console=False,
        enable_file=True,
        enable_rich_formatting=False,
        enable_performance_monitoring=True,
        enable_resource_monitoring=True,
        timeout_threshold=600.0,  # Longer timeout for production
        execution_time_threshold=60.0,  # Less sensitive to slow execution
        auto_cleanup=True,
        cleanup_interval_minutes=30,
        max_error_history=100,
        max_performance_history=100
    )


def get_testing_config() -> AigieConfig:
    """Get configuration optimized for testing."""
    return AigieConfig(
        log_level="WARNING",
        enable_console=False,
        enable_file=False,
        enable_rich_formatting=False,
        enable_performance_monitoring=False,
        enable_resource_monitoring=False,
        timeout_threshold=30.0,
        auto_cleanup=True,
        cleanup_interval_minutes=5
    )
