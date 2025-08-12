#!/usr/bin/env python3
"""
Jukeyman Research Swarm - Command Line Interface
By Rick Jefferson Solutions
Provides easy access to research capabilities with flexible configuration
"""

import asyncio
import sys
import os
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

# Import our main research system
from autonomous_research_swarm import main as research_main, Config, API_KEYS

console = Console()

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        console.print(f"[yellow]Warning: Config file {config_path} not found. Using defaults.[/yellow]")
        return {}

def display_banner():
    """Display the application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              🎵 JUKEYMAN RESEARCH SWARM 🎵                  ║
    ║                By Rick Jefferson Solutions                   ║
    ║    AI-Powered Research Assistant with Multi-Agent System     ║
    ║         Comprehensive • Collaborative • Intelligent          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")

def display_api_status():
    """Display status of configured API keys"""
    table = Table(title="🔑 API Configuration Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Key Preview", style="dim")
    
    for key_name, key_value in API_KEYS.items():
        if key_value and len(key_value) > 8:
            preview = f"{key_value[:4]}...{key_value[-4:]}"
            status = "✅ Configured"
        else:
            preview = "Not set"
            status = "❌ Missing"
        
        service_name = key_name.replace('_', ' ').title()
        table.add_row(service_name, status, preview)
    
    console.print(table)

def display_research_topics():
    """Display example research topics"""
    topics = [
        "Impact of Large Language Models on healthcare workflows (2020-2025)",
        "Quantum computing applications in cryptography and security",
        "Climate change effects on global food security and agriculture",
        "Artificial intelligence in autonomous vehicle safety systems",
        "Blockchain technology adoption in supply chain management",
        "Gene therapy breakthroughs in treating rare diseases",
        "Renewable energy storage solutions and grid integration",
        "Social media impact on mental health in adolescents",
        "Space exploration technologies and Mars colonization prospects",
        "Cybersecurity threats in IoT and smart city infrastructure"
    ]
    
    console.print("\n[bold]📋 Example Research Topics:[/bold]")
    for i, topic in enumerate(topics, 1):
        console.print(f"  {i:2d}. {topic}")

async def run_research_with_progress(topic: str, config_overrides: Dict[str, Any] = None):
    """Run research with a progress indicator"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("🔍 Initializing research swarm...", total=None)
        
        try:
            # Update progress descriptions during research
            progress.update(task, description="🤖 Starting autonomous agents...")
            await asyncio.sleep(1)
            
            progress.update(task, description="📊 Planning research strategy...")
            await asyncio.sleep(1)
            
            progress.update(task, description="🌐 Conducting web research...")
            result = await research_main(topic, config_overrides)
            
            progress.update(task, description="✅ Research complete!", completed=True)
            return result
            
        except Exception as e:
            progress.update(task, description=f"❌ Error: {str(e)}", completed=True)
            raise

def display_results(result: Dict[str, Any]):
    """Display research results in a formatted way"""
    console.print("\n" + "="*60)
    console.print("[bold green]🎉 RESEARCH COMPLETED SUCCESSFULLY![/bold green]")
    console.print("="*60)
    
    # Results summary
    summary_table = Table(title="📊 Research Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Final Quality Score", f"{result.get('score', 0):.3f}")
    summary_table.add_row("Evidence Sources", str(result.get('evidence_count', 0)))
    summary_table.add_row("Output Directory", result.get('run_dir', 'Unknown'))
    
    console.print(summary_table)
    
    # File locations
    run_dir = result.get('run_dir', '')
    if run_dir:
        console.print("\n[bold]📁 Generated Files:[/bold]")
        files = [
            ("📄 Research Report", os.path.join(run_dir, "report.md")),
            ("🔍 Evidence Database", os.path.join(run_dir, "evidence.jsonl")),
            ("📋 Research Plan", os.path.join(run_dir, "plan.json")),
            ("📊 Event Log", os.path.join(run_dir, "events.jsonl"))
        ]
        
        for desc, filepath in files:
            if os.path.exists(filepath):
                console.print(f"  {desc}: [link]{filepath}[/link]")
            else:
                console.print(f"  {desc}: [dim]Not found[/dim]")

def preview_report(run_dir: str):
    """Preview the generated report"""
    report_path = os.path.join(run_dir, "report.md")
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Show first 1000 characters
        preview = content[:1000] + "..." if len(content) > 1000 else content
        
        console.print("\n[bold]📖 Report Preview:[/bold]")
        console.print(Panel(Markdown(preview), title="Research Report", border_style="blue"))
    else:
        console.print("[red]Report file not found.[/red]")

def interactive_mode():
    """Run in interactive mode for topic selection"""
    console.print("\n[bold]🎯 Interactive Research Mode[/bold]")
    console.print("Enter your research topic or select from examples below:")
    
    display_research_topics()
    
    while True:
        try:
            user_input = console.input("\n[bold]Enter topic (or number 1-10, 'q' to quit): [/bold]")
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                console.print("[yellow]Goodbye![/yellow]")
                return None
            
            # Check if it's a number selection
            if user_input.isdigit():
                num = int(user_input)
                if 1 <= num <= 10:
                    topics = [
                        "Impact of Large Language Models on healthcare workflows (2020-2025)",
                        "Quantum computing applications in cryptography and security",
                        "Climate change effects on global food security and agriculture",
                        "Artificial intelligence in autonomous vehicle safety systems",
                        "Blockchain technology adoption in supply chain management",
                        "Gene therapy breakthroughs in treating rare diseases",
                        "Renewable energy storage solutions and grid integration",
                        "Social media impact on mental health in adolescents",
                        "Space exploration technologies and Mars colonization prospects",
                        "Cybersecurity threats in IoT and smart city infrastructure"
                    ]
                    return topics[num - 1]
                else:
                    console.print("[red]Please enter a number between 1-10.[/red]")
                    continue
            
            # Custom topic
            if len(user_input.strip()) > 10:
                return user_input.strip()
            else:
                console.print("[red]Please enter a more detailed research topic (at least 10 characters).[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Jukeyman Research Swarm - AI-Powered Research Assistant by Rick Jefferson Solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "Impact of AI on healthcare"
  python cli.py --interactive
  python cli.py --config custom_config.yaml "Climate change research"
  python cli.py --max-steps 15 --quality-target 0.9 "Quantum computing"
        """
    )
    
    parser.add_argument(
        "topic",
        nargs="*",
        help="Research topic (if not provided, will use interactive mode)"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode for topic selection"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum research steps (overrides config)"
    )
    
    parser.add_argument(
        "--quality-target",
        type=float,
        help="Quality target threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "--coverage-target",
        type=float,
        help="Coverage target threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "--parallel-researchers",
        type=int,
        help="Number of parallel researchers"
    )
    
    parser.add_argument(
        "--provider",
        choices=["perplexity", "google", "moonshot"],
        help="LLM provider to use"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the report after completion"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show API configuration status and exit"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Display banner unless in quiet mode
    if not args.quiet:
        display_banner()
    
    # Show status and exit if requested
    if args.status:
        display_api_status()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Build config overrides from command line arguments
    config_overrides = {}
    if args.max_steps:
        config_overrides['max_steps'] = args.max_steps
    if args.quality_target:
        config_overrides['quality_target'] = args.quality_target
    if args.coverage_target:
        config_overrides['coverage_target'] = args.coverage_target
    if args.parallel_researchers:
        config_overrides['parallel_researchers'] = args.parallel_researchers
    if args.provider:
        config_overrides['llm_provider'] = args.provider
    
    # Determine research topic
    if args.interactive or not args.topic:
        topic = interactive_mode()
        if not topic:
            return
    else:
        topic = " ".join(args.topic)
    
    if not topic or len(topic.strip()) < 5:
        console.print("[red]Error: Please provide a valid research topic.[/red]")
        return
    
    # Display research info
    if not args.quiet:
        console.print(f"\n[bold]🎯 Research Topic:[/bold] {topic}")
        console.print(f"[bold]⚙️  Configuration:[/bold] {args.config}")
        if config_overrides:
            console.print(f"[bold]🔧 Overrides:[/bold] {config_overrides}")
    
    try:
        # Run the research
        if args.quiet:
            result = await research_main(topic, config_overrides)
        else:
            result = await run_research_with_progress(topic, config_overrides)
        
        # Display results
        if not args.quiet:
            display_results(result)
            
            if args.preview:
                preview_report(result.get('run_dir', ''))
        else:
            # Quiet mode - just print the run directory
            print(result.get('run_dir', ''))
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during research: {e}[/red]")
        if not args.quiet:
            console.print("\n[dim]Use --help for usage information.[/dim]")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)