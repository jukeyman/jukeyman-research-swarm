#!/usr/bin/env python3
"""
Jukeyman Research Swarm - Test Script
By Rick Jefferson Solutions
Demonstrates functionality and validates the system
"""

import asyncio
import os
import json
import time
from pathlib import Path

# Import the research system
from autonomous_research_swarm import main as research_main, API_KEYS

def test_api_keys():
    """Test that API keys are properly configured"""
    print("🔑 Testing API Key Configuration...")
    
    required_keys = [
        'PERPLEXITY_API_KEY',
        'GOOGLE_AI_API_KEY', 
        'HUGGINGFACE_TOKEN'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in API_KEYS or not API_KEYS[key]:
            missing_keys.append(key)
        else:
            print(f"  ✅ {key}: {API_KEYS[key][:8]}...")
    
    if missing_keys:
        print(f"  ❌ Missing keys: {missing_keys}")
        return False
    
    print(f"  ✅ All {len(API_KEYS)} API keys configured")
    return True

async def test_basic_research():
    """Test basic research functionality"""
    print("\n🔬 Testing Basic Research Functionality...")
    
    # Simple test topic
    test_topic = "Artificial intelligence applications in healthcare"
    
    # Minimal configuration for quick test
    config_overrides = {
        'max_steps': 3,
        'parallel_researchers': 2,
        'quality_target': 0.6,
        'coverage_target': 0.6
    }
    
    try:
        print(f"  📋 Topic: {test_topic}")
        print(f"  ⚙️ Config: {config_overrides}")
        
        start_time = time.time()
        result = await research_main(test_topic, config_overrides)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"  ⏱️ Duration: {duration:.1f} seconds")
        print(f"  📊 Score: {result.get('score', 0):.3f}")
        print(f"  📄 Evidence: {result.get('evidence_count', 0)} sources")
        print(f"  📁 Output: {result.get('run_dir', 'Unknown')}")
        
        # Validate output files
        run_dir = result.get('run_dir', '')
        if run_dir and os.path.exists(run_dir):
            files_to_check = ['report.md', 'evidence.jsonl', 'plan.json', 'events.jsonl']
            for filename in files_to_check:
                filepath = os.path.join(run_dir, filename)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"    ✅ {filename}: {size} bytes")
                else:
                    print(f"    ❌ {filename}: Missing")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_llm_providers():
    """Test different LLM providers"""
    print("\n🤖 Testing LLM Providers...")
    
    from autonomous_research_swarm import call_llm
    
    providers = ['perplexity', 'google', 'moonshot']
    test_prompt = "What is artificial intelligence? Respond in one sentence."
    
    for provider in providers:
        try:
            print(f"  Testing {provider}...")
            response = await call_llm(test_prompt, provider=provider)
            print(f"    ✅ {provider}: {response[:100]}...")
        except Exception as e:
            print(f"    ❌ {provider}: {str(e)[:100]}...")

async def test_search_functionality():
    """Test search capabilities"""
    print("\n🔍 Testing Search Functionality...")
    
    from autonomous_research_swarm import web_search
    
    try:
        query = "machine learning applications"
        results = await web_search(query, k=3)
        
        print(f"  📋 Query: {query}")
        print(f"  📊 Results: {len(results)}")
        
        for i, result in enumerate(results[:3], 1):
            title = result.get('title', 'No title')[:50]
            url = result.get('url', 'No URL')[:50]
            print(f"    {i}. {title}... ({url}...)")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"  ❌ Search error: {e}")
        return False

def test_output_analysis(run_dir: str):
    """Analyze the quality of research output"""
    print("\n📊 Analyzing Research Output...")
    
    if not run_dir or not os.path.exists(run_dir):
        print("  ❌ No output directory found")
        return False
    
    # Check report quality
    report_path = os.path.join(run_dir, 'report.md')
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        print(f"  📄 Report length: {len(report_content)} characters")
        print(f"  📝 Report lines: {len(report_content.splitlines())}")
        
        # Check for key sections
        sections = ['summary', 'findings', 'conclusion', 'source']
        found_sections = []
        for section in sections:
            if section.lower() in report_content.lower():
                found_sections.append(section)
        
        print(f"  📋 Sections found: {found_sections}")
    
    # Check evidence quality
    evidence_path = os.path.join(run_dir, 'evidence.jsonl')
    if os.path.exists(evidence_path):
        evidence_count = 0
        unique_domains = set()
        
        with open(evidence_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        evidence = json.loads(line)
                        evidence_count += 1
                        if 'domain' in evidence:
                            unique_domains.add(evidence['domain'])
                    except json.JSONDecodeError:
                        pass
        
        print(f"  🔍 Evidence items: {evidence_count}")
        print(f"  🌐 Unique domains: {len(unique_domains)}")
        print(f"  📊 Top domains: {list(unique_domains)[:5]}")
    
    return True

async def run_comprehensive_test():
    """Run a comprehensive test of the research system"""
    print("🧪 JUKEYMAN RESEARCH SWARM - COMPREHENSIVE TEST")
    print("   By Rick Jefferson Solutions")
    print("=" * 60)
    
    # Test 1: API Keys
    api_test = test_api_keys()
    
    # Test 2: LLM Providers
    await test_llm_providers()
    
    # Test 3: Search Functionality
    search_test = await test_search_functionality()
    
    # Test 4: Basic Research
    research_result = None
    if api_test and search_test:
        research_result = await test_basic_research()
    else:
        print("\n⚠️ Skipping research test due to API/search issues")
    
    # Test 5: Output Analysis
    if research_result:
        # Get the latest run directory
        runs_dir = 'runs'
        if os.path.exists(runs_dir):
            run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if run_dirs:
                latest_run = max(run_dirs)
                latest_run_path = os.path.join(runs_dir, latest_run)
                test_output_analysis(latest_run_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("API Configuration", api_test),
        ("Search Functionality", search_test),
        ("Research Execution", research_result)
    ]
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED! The system is ready for use.")
        print("\n🚀 Try running: python cli.py --interactive")
    else:
        print("\n⚠️ Some tests failed. Check the configuration and API keys.")
    
    return passed == len(tests)

async def quick_demo():
    """Run a quick demonstration"""
    print("🎬 JUKEYMAN RESEARCH SWARM - QUICK DEMO")
    print("   By Rick Jefferson Solutions")
    print("=" * 40)
    
    demo_topic = "Benefits of renewable energy"
    
    config = {
        'max_steps': 2,
        'parallel_researchers': 1,
        'quality_target': 0.5,
        'coverage_target': 0.5
    }
    
    print(f"📋 Demo Topic: {demo_topic}")
    print("⏱️ Running quick research (should take ~30 seconds)...")
    
    try:
        result = await research_main(demo_topic, config)
        
        print(f"\n✅ Demo completed!")
        print(f"📊 Score: {result.get('score', 0):.3f}")
        print(f"📁 Results: {result.get('run_dir', 'Unknown')}")
        
        # Show a snippet of the report
        run_dir = result.get('run_dir', '')
        if run_dir:
            report_path = os.path.join(run_dir, 'report.md')
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print("\n📖 Report Preview:")
                print("-" * 40)
                print(content[:500] + "..." if len(content) > 500 else content)
                print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Quick demo mode
        asyncio.run(quick_demo())
    else:
        # Comprehensive test mode
        asyncio.run(run_comprehensive_test())