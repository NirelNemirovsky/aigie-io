"""
Command-line interface for Aigie.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from .auto_integration import auto_integrate, stop_integration, get_status, get_analysis
from .utils.config import AigieConfig, get_development_config, get_production_config, get_testing_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Aigie - AI Agent Runtime Error Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enable monitoring with default settings
  aigie enable
  
  # Enable monitoring with custom config
  aigie enable --config production
  
  # Show current status
  aigie status
  
  # Show detailed analysis
  aigie analysis
  
  # Stop monitoring
  aigie disable
  
  # Generate configuration file
  aigie config --generate config.yml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Enable command
    enable_parser = subparsers.add_parser('enable', help='Enable Aigie monitoring')
    enable_parser.add_argument('--config', choices=['development', 'production', 'testing', 'custom'],
                              default='development', help='Configuration preset')
    enable_parser.add_argument('--config-file', type=str, help='Path to custom configuration file')
    enable_parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                              help='Log level override')
    enable_parser.add_argument('--enable-console', action='store_true', help='Enable console output')
    enable_parser.add_argument('--enable-file', action='store_true', help='Enable file logging')
    enable_parser.add_argument('--log-file', type=str, help='Log file path')
    
    # Gemini options
    enable_parser.add_argument('--gemini-project', type=str, help='Google Cloud project ID for Gemini')
    enable_parser.add_argument('--gemini-location', type=str, default='us-central1', help='Gemini location')
    enable_parser.add_argument('--enable-gemini', action='store_true', help='Enable Gemini-powered error analysis')
    enable_parser.add_argument('--disable-gemini', action='store_true', help='Disable Gemini-powered error analysis')
    
    # Disable command
    subparsers.add_parser('disable', help='Disable Aigie monitoring')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show monitoring status')
    status_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    status_parser.add_argument('--detailed', action='store_true', help='Show detailed status')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analysis', help='Show detailed analysis')
    analysis_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    analysis_parser.add_argument('--window', type=int, default=60, help='Time window in minutes')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--generate', type=str, help='Generate configuration file')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--validate', type=str, help='Validate configuration file')
    
    # Gemini command
    gemini_parser = subparsers.add_parser('gemini', help='Gemini integration management')
    gemini_parser.add_argument('--setup', type=str, help='Setup Gemini with project ID')
    gemini_parser.add_argument('--status', action='store_true', help='Show Gemini status')
    gemini_parser.add_argument('--test', action='store_true', help='Test Gemini connection')
    
    # Version command
    subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'enable':
            return handle_enable(args)
        elif args.command == 'disable':
            return handle_disable(args)
        elif args.command == 'status':
            return handle_status(args)
        elif args.command == 'analysis':
            return handle_analysis(args)
        elif args.command == 'config':
            return handle_config(args)
        elif args.command == 'gemini':
            return handle_gemini(args)
        elif args.command == 'version':
            return handle_version(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def handle_enable(args) -> int:
    """Handle the enable command."""
    print("🚀 Enabling Aigie monitoring...")
    
    # Get configuration
    config = get_configuration(args)
    
    # Enable monitoring
    try:
        integrator = auto_integrate(config)
        print("✅ Aigie monitoring enabled successfully!")
        print(f"📊 Log level: {config.log_level}")
        print(f"🖥️  Console output: {'Enabled' if config.enable_console else 'Disabled'}")
        print(f"📁 File logging: {'Enabled' if config.enable_file else 'Disabled'}")
        if config.enable_file and config.log_file_path:
            print(f"📄 Log file: {config.log_file_path}")
        return 0
    except Exception as e:
        print(f"❌ Failed to enable monitoring: {e}")
        return 1


def handle_disable(args) -> int:
    """Handle the disable command."""
    print("🛑 Disabling Aigie monitoring...")
    
    try:
        stop_integration()
        print("✅ Aigie monitoring disabled successfully!")
        return 0
    except Exception as e:
        print(f"❌ Failed to disable monitoring: {e}")
        return 1


def handle_status(args) -> int:
    """Handle the status command."""
    status = get_status()
    
    if not status:
        print("❌ Aigie monitoring is not enabled")
        return 1
    
    if args.json:
        print(json.dumps(status, indent=2))
        return 0
    
    # Display status in human-readable format
    print("📊 Aigie Monitoring Status")
    print("=" * 40)
    
    # Basic status
    print(f"🔄 Status: {'Active' if status['is_integrated'] else 'Inactive'}")
    if status['integration_start_time']:
        print(f"⏰ Started: {status['integration_start_time']}")
    
    # Error detection status
    error_detector = status.get('error_detector', {})
    print(f"🚨 Error Detection: {'Active' if error_detector.get('is_monitoring') else 'Inactive'}")
    print(f"📈 Total Errors: {error_detector.get('total_errors', 0)}")
    
    # Recent errors
    recent_errors = error_detector.get('recent_errors', {})
    if recent_errors.get('total_errors', 0) > 0:
        print(f"⚡ Recent Errors (5min): {recent_errors['total_errors']}")
        
        if args.detailed and recent_errors.get('severity_distribution'):
            print("\nSeverity Distribution:")
            for severity, count in recent_errors['severity_distribution'].items():
                print(f"  {severity.upper()}: {count}")
    
    # Framework status
    langchain_status = status.get('langchain', {})
    print(f"🔗 LangChain: {'Active' if langchain_status.get('is_intercepting') else 'Inactive'}")
    
    langgraph_status = status.get('langgraph', {})
    print(f"🕸️  LangGraph: {'Active' if langgraph_status.get('is_intercepting') else 'Inactive'}")
    
    # System health
    system_health = status.get('system_health', {})
    if system_health:
        print(f"\n🏥 System Health: {system_health.get('overall_status', 'Unknown')}")
        
        if args.detailed:
            memory_health = system_health.get('memory', {})
            if memory_health:
                print(f"  💾 Memory: {memory_health.get('status', 'Unknown')} ({memory_health.get('usage_percent', 0):.1f}%)")
            
            cpu_health = system_health.get('cpu', {})
            if cpu_health:
                print(f"  🖥️  CPU: {cpu_health.get('status', 'Unknown')} ({cpu_health.get('usage_percent', 0):.1f}%)")
    
    return 0


def handle_analysis(args) -> int:
    """Handle the analysis command."""
    analysis = get_analysis()
    
    if not analysis:
        print("❌ Aigie monitoring is not enabled")
        return 1
    
    if args.json:
        print(json.dumps(analysis, indent=2))
        return 0
    
    # Display analysis in human-readable format
    print("📊 Aigie Detailed Analysis")
    print("=" * 40)
    
    # Error summary
    error_summary = analysis.get('error_summary', {})
    if error_summary:
        print(f"🚨 Total Errors: {error_summary.get('total_errors', 0)}")
        
        if error_summary.get('severity_distribution'):
            print("\nSeverity Distribution:")
            for severity, count in error_summary['severity_distribution'].items():
                print(f"  {severity.upper()}: {count}")
        
        if error_summary.get('type_distribution'):
            print("\nError Type Distribution:")
            for error_type, count in error_summary['type_distribution'].items():
                print(f"  {error_type}: {count}")
    
    # Performance summary
    performance_summary = analysis.get('performance_summary', {})
    if performance_summary:
        print(f"\n📈 Performance Summary (Last {args.window} minutes):")
        print(f"  Total Executions: {performance_summary.get('total_executions', 0)}")
        print(f"  Avg Execution Time: {performance_summary.get('avg_execution_time', 0):.2f}s")
        print(f"  Max Execution Time: {performance_summary.get('max_execution_time', 0):.2f}s")
        print(f"  Min Execution Time: {performance_summary.get('min_execution_time', 0):.2f}s")
    
    # LangGraph analysis
    langgraph_analysis = analysis.get('langgraph_analysis', {})
    if langgraph_analysis:
        graph_analysis = langgraph_analysis.get('graph_analysis', {})
        if graph_analysis:
            print(f"\n🕸️  Graph Analysis:")
            print(f"  Total Graphs: {graph_analysis.get('total_graphs', 0)}")
        
        node_stats = langgraph_analysis.get('node_execution_stats', {})
        if node_stats:
            print(f"\n🔗 Node Execution Statistics:")
            print(f"  Total Nodes: {node_stats.get('total_nodes', 0)}")
            
            for node_name, stats in node_stats.get('nodes', {}).items():
                success_rate = stats.get('success_rate', 0)
                print(f"  {node_name}: {stats.get('total_executions', 0)} executions, {success_rate:.1f}% success rate")
    
    return 0


def handle_config(args) -> int:
    """Handle the config command."""
    if args.generate:
        return generate_config_file(args.generate)
    elif args.show:
        return show_current_config()
    elif args.validate:
        return validate_config_file(args.validate)
    else:
        print("Please specify an action: --generate, --show, or --validate")
        return 1


def handle_gemini(args) -> int:
    """Handle the Gemini command."""
    if args.setup:
        return setup_gemini(args.setup)
    elif args.status:
        return show_gemini_status()
    elif args.test:
        return test_gemini_connection()
    else:
        print("Please specify an action: --setup, --status, or --test")
        return 1


def handle_version(args) -> int:
    """Handle the version command."""
    try:
        from . import __version__
        print(f"Aigie version {__version__}")
        return 0
    except ImportError:
        print("Aigie version unknown")
        return 1


def get_configuration(args) -> AigieConfig:
    """Get configuration based on command line arguments."""
    if args.config_file:
        # Load from custom file
        config = AigieConfig.from_file(args.config_file)
    else:
        # Use preset configuration
        if args.config == 'development':
            config = get_development_config()
        elif args.config == 'production':
            config = get_production_config()
        elif args.config == 'testing':
            config = get_testing_config()
        else:
            config = AigieConfig()
    
    # Override with command line arguments
    if args.log_level:
        config.log_level = args.log_level
    
    if args.enable_console is not None:
        config.enable_console = args.enable_console
    
    if args.enable_file is not None:
        config.enable_file = args.enable_file
    
    if args.log_file:
        config.log_file_path = args.log_file
        config.enable_file = True
    
    return config


def generate_config_file(file_path: str) -> int:
    """Generate a configuration file."""
    try:
        # Create development config as template
        config = get_development_config()
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config.save_to_file(file_path)
        print(f"✅ Configuration file generated: {file_path}")
        print("📝 Edit the file to customize settings, then use --config-file to load it")
        return 0
    except Exception as e:
        print(f"❌ Failed to generate configuration file: {e}")
        return 1


def show_current_config() -> int:
    """Show current configuration."""
    try:
        integrator = get_status()
        if not integrator:
            print("❌ Aigie monitoring is not enabled")
            return 1
        
        config = integrator.get('configuration', {})
        if config:
            print("📋 Current Configuration:")
            print(json.dumps(config, indent=2))
        else:
            print("No configuration information available")
        return 0
    except Exception as e:
        print(f"❌ Failed to show configuration: {e}")
        return 1


def validate_config_file(file_path: str) -> int:
    """Validate a configuration file."""
    try:
        config = AigieConfig.from_file(file_path)
        print(f"✅ Configuration file is valid: {file_path}")
        print("📋 Configuration summary:")
        print(f"  Log level: {config.log_level}")
        print(f"  Console output: {'Enabled' if config.enable_console else 'Disabled'}")
        print(f"  File logging: {'Enabled' if config.enable_file else 'Disabled'}")
        print(f"  Performance monitoring: {'Enabled' if config.enable_performance_monitoring else 'Disabled'}")
        print(f"  Resource monitoring: {'Enabled' if config.enable_resource_monitoring else 'Disabled'}")
        return 0
    except Exception as e:
        print(f"❌ Configuration file is invalid: {e}")
        return 1


def setup_gemini(project_id: str) -> int:
    """Setup Gemini integration with project ID."""
    try:
        print(f"🚀 Setting up Gemini integration for project: {project_id}")
        
        # Check if gcloud is available
        import subprocess
        try:
            result = subprocess.run(['gcloud', 'auth', 'list'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Google Cloud SDK not available or not authenticated")
                print("Please install and authenticate with: gcloud auth login")
                return 1
        except FileNotFoundError:
            print("❌ Google Cloud SDK not found")
            print("Please install Google Cloud SDK first")
            return 1
        
        # Set project
        try:
            result = subprocess.run(['gcloud', 'config', 'set', 'project', project_id], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed to set project: {result.stderr}")
                return 1
            print(f"✅ Project set to: {project_id}")
        except Exception as e:
            print(f"❌ Failed to set project: {e}")
            return 1
        
        # Enable Vertex AI API
        try:
            print("🔧 Enabling Vertex AI API...")
            result = subprocess.run(['gcloud', 'services', 'enable', 'aiplatform.googleapis.com'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️  Warning: Failed to enable Vertex AI API: {result.stderr}")
                print("You may need to enable it manually in the Google Cloud Console")
            else:
                print("✅ Vertex AI API enabled")
        except Exception as e:
            print(f"⚠️  Warning: Failed to enable Vertex AI API: {e}")
        
        # Create configuration
        config = AigieConfig()
        config.gemini_project_id = project_id
        config.enable_gemini_analysis = True
        
        # Save configuration
        config.save_to_file("aigie_gemini_config.json")
        print("✅ Gemini configuration saved to aigie_gemini_config.json")
        
        print("\n📋 Next steps:")
        print("1. Use this config file: aigie enable --config-file aigie_gemini_config.json")
        print("2. Or set environment variables:")
        print(f"   export GOOGLE_CLOUD_PROJECT={project_id}")
        print("   export AIGIE_ENABLE_GEMINI=true")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to setup Gemini: {e}")
        return 1


def show_gemini_status() -> int:
    """Show Gemini integration status."""
    try:
        integrator = get_status()
        if not integrator:
            print("❌ Aigie monitoring is not enabled")
            return 1
        
        gemini_status = integrator.get('system_health', {}).get('gemini_status', {})
        if gemini_status:
            print("🤖 Gemini Integration Status:")
            print(f"  Enabled: {'Yes' if gemini_status.get('enabled') else 'No'}")
            print(f"  Project ID: {gemini_status.get('project_id', 'N/A')}")
            print(f"  Location: {gemini_status.get('location', 'N/A')}")
            print(f"  Model Loaded: {'Yes' if gemini_status.get('model_loaded') else 'No'}")
            
            if not gemini_status.get('enabled'):
                print(f"  Reason: {gemini_status.get('reason', 'Unknown')}")
        else:
            print("🤖 Gemini integration not available")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to show Gemini status: {e}")
        return 1


def test_gemini_connection() -> int:
    """Test Gemini connection."""
    try:
        print("🧪 Testing Gemini connection...")
        
        # Try to create a Gemini analyzer
        from .core.ai.gemini_analyzer import GeminiAnalyzer
        
        # Get project ID from environment or config
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            print("❌ No project ID found. Set GOOGLE_CLOUD_PROJECT environment variable")
            return 1
        
        analyzer = GeminiAnalyzer(project_id)
        
        if analyzer.is_available():
            print("✅ Gemini connection successful!")
            print(f"  Project: {project_id}")
            print(f"  Location: {analyzer.location}")
            print(f"  Model: Available")
        else:
            print("❌ Gemini connection failed")
            print("Please check:")
            print("1. Project ID is correct")
            print("2. Authentication is set up: gcloud auth application-default login")
            print("3. Vertex AI API is enabled")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to test Gemini connection: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
