#!/usr/bin/env python3
"""
Jukeyman Research Swarm - Setup Script
By Rick Jefferson Solutions
Helps users configure and validate their environment
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_banner():
    """Display setup banner"""
    print("\n" + "=" * 60)
    print("🎵 JUKEYMAN RESEARCH SWARM - SETUP")
    print("   By Rick Jefferson Solutions")
    print("=" * 60)
    print("This script will help you set up the research environment.")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ❌ Python {version.major}.{version.minor} detected")
        print("  ⚠️ Python 3.8+ is required")
        return False
    
    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"  ❌ {requirements_file} not found")
        return False
    
    try:
        print("  📥 Installing packages (this may take a few minutes)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            capture_output=True,
            text=True,
            check=True
        )
        print("  ✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Installation failed: {e}")
        print(f"  📝 Error output: {e.stderr}")
        return False

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 Checking API key configuration...")
    
    json_file = "Untitled-1.json"
    if not os.path.exists(json_file):
        print(f"  ❌ {json_file} not found")
        print("  💡 Create this file with your API keys")
        return False
    
    try:
        with open(json_file, 'r') as f:
            keys = json.load(f)
        
        required_keys = [
            'PERPLEXITY_API_KEY',
            'GOOGLE_AI_API_KEY',
            'HUGGINGFACE_TOKEN'
        ]
        
        missing_keys = []
        configured_keys = []
        
        for key in required_keys:
            if key in keys and keys[key]:
                configured_keys.append(key)
                print(f"  ✅ {key}: Configured")
            else:
                missing_keys.append(key)
                print(f"  ❌ {key}: Missing")
        
        # Check optional keys
        optional_keys = [
            'KAGGLE_USERNAME', 'KAGGLE_KEY',
            'FIRECRAWL_API_KEY', 'MOONSHOT_API_KEY'
        ]
        
        for key in optional_keys:
            if key in keys and keys[key]:
                configured_keys.append(key)
                print(f"  ℹ️ {key}: Configured (optional)")
        
        print(f"\n  📊 Summary: {len(configured_keys)} keys configured, {len(missing_keys)} required keys missing")
        
        if missing_keys:
            print(f"\n  ⚠️ Missing required keys: {missing_keys}")
            print("  💡 Add these to your Untitled-1.json file")
            return False
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON format: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error reading keys: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ['runs', 'cache', 'logs']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✅ {directory}/")
        except Exception as e:
            print(f"  ❌ Failed to create {directory}/: {e}")
            return False
    
    return True

def validate_config():
    """Validate configuration file"""
    print("\n⚙️ Validating configuration...")
    
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        print(f"  ❌ {config_file} not found")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check essential sections
        required_sections = ['loop', 'llm', 'search', 'safety', 'budget']
        missing_sections = []
        
        for section in required_sections:
            if section in config:
                print(f"  ✅ {section}: Configured")
            else:
                missing_sections.append(section)
                print(f"  ❌ {section}: Missing")
        
        if missing_sections:
            print(f"  ⚠️ Missing sections: {missing_sections}")
            return False
        
        print("  ✅ Configuration is valid")
        return True
        
    except ImportError:
        print("  ⚠️ PyYAML not installed, skipping config validation")
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def run_quick_test():
    """Run a quick system test"""
    print("\n🧪 Running quick system test...")
    
    try:
        # Import the main module to check for import errors
        import autonomous_research_swarm
        print("  ✅ Main module imports successfully")
        
        # Check if API keys are loaded
        if hasattr(autonomous_research_swarm, 'API_KEYS'):
            key_count = len(autonomous_research_swarm.API_KEYS)
            print(f"  ✅ {key_count} API keys loaded")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("  💡 Check that all dependencies are installed")
        return False
    except Exception as e:
        print(f"  ❌ System test failed: {e}")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS")
    print("=" * 60)
    
    print("\n1. 🧪 Run comprehensive tests:")
    print("   python test_research.py")
    
    print("\n2. 🎬 Try a quick demo:")
    print("   python test_research.py --demo")
    
    print("\n3. 🚀 Start interactive research:")
    print("   python cli.py --interactive")
    
    print("\n4. 📋 Run specific research:")
    print("   python cli.py --topic \"Your research topic\"")
    
    print("\n5. 📖 Read the documentation:")
    print("   Check README.md for detailed usage")
    
    print("\n" + "=" * 60)
    print("🎉 Jukeyman Research Swarm Setup Complete!")
    print("   By Rick Jefferson Solutions")
    print("   Happy researching!")
    print("=" * 60)

def main():
    """Main setup function"""
    print_banner()
    
    # Run setup checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("API Keys", check_api_keys),
        ("Directories", create_directories),
        ("Configuration", validate_config),
        ("System Test", run_quick_test)
    ]
    
    passed = 0
    failed = []
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                failed.append(check_name)
        except Exception as e:
            print(f"  ❌ {check_name} check failed with error: {e}")
            failed.append(check_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)
    
    print(f"✅ Passed: {passed}/{len(checks)} checks")
    
    if failed:
        print(f"❌ Failed: {failed}")
        print("\n⚠️ Please fix the failed checks before proceeding.")
        
        # Provide specific help for common issues
        if "API Keys" in failed:
            print("\n💡 API Key Help:")
            print("   - Get Perplexity API key: https://www.perplexity.ai/settings/api")
            print("   - Get Google AI key: https://aistudio.google.com/app/apikey")
            print("   - Get Hugging Face token: https://huggingface.co/settings/tokens")
        
        if "Dependencies" in failed:
            print("\n💡 Dependency Help:")
            print("   - Try: pip install --upgrade pip")
            print("   - Try: pip install -r requirements.txt --no-cache-dir")
        
        return False
    else:
        show_next_steps()
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)