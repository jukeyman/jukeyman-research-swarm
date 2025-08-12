# 🎵 Jukeyman Research Swarm

> **A production-ready AI research assistant that coordinates multiple agents to conduct comprehensive, evidence-based research on any topic.**
> **By Rick Jefferson Solutions**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Features

### 🎯 **Intelligent Multi-Agent System**
- **Orchestrator**: Manages the research loop and convergence criteria
- **Planner**: Decomposes topics into atomic research questions
- **Researchers**: Execute parallel web searches and content extraction
- **Analyst**: Synthesizes findings into structured reports
- **Critic**: Evaluates quality and identifies research gaps
- **Librarian**: Deduplicates and ranks evidence by authority

### 🔍 **Advanced Research Capabilities**
- **Autonomous Planning**: Breaks down complex topics into research subtasks
- **Parallel Processing**: Multiple researchers work simultaneously
- **Quality Convergence**: Continues until quality and coverage thresholds are met
- **Evidence Ranking**: Prioritizes authoritative sources (.edu, .gov, .org)
- **Citation Management**: Full traceability with source URLs and timestamps
- **Gap Detection**: Identifies missing information and replans accordingly

### 🛡️ **Safety & Quality Controls**
- **Domain Filtering**: Configurable allow/deny lists for source domains
- **Robots.txt Compliance**: Respects website crawling policies
- **Rate Limiting**: Prevents API abuse and respects service limits
- **Content Validation**: Filters out low-quality and irrelevant content
- **Budget Management**: Token, time, and request limits

### 🔌 **Multi-Provider Integration**
- **LLM Providers**: Perplexity, Google AI (Gemini), Moonshot, Kimi
- **Search Engines**: Perplexity browsing, Firecrawl, HyperBrowser
- **Embeddings**: Hugging Face, Google AI
- **Data Sources**: Kaggle, Papers with Code, academic databases

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/jukeyman/jukeyman-research-swarm.git
cd jukeyman-research-swarm

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Setup

The system comes pre-configured with API keys for immediate use. All keys are already integrated:

- ✅ **Perplexity AI**: For LLM and search capabilities
- ✅ **Google AI**: Gemini models for analysis
- ✅ **Moonshot AI**: Alternative LLM provider
- ✅ **Hugging Face**: Embeddings and model access
- ✅ **Firecrawl**: Web scraping and content extraction
- ✅ **Kaggle**: Dataset access
- ✅ **Papers with Code**: Academic paper access

### 3. Run Your First Research

```bash
# Simple command-line usage
python autonomous_research_swarm.py "Impact of AI on healthcare workflows"

# Or use the interactive CLI
python cli.py --interactive

# Check API status
python cli.py --status
```

## 📋 Usage Examples

### Command Line Interface

```bash
# Basic research
python cli.py "Climate change effects on agriculture"

# Interactive mode with topic selection
python cli.py --interactive

# Custom configuration
python cli.py --max-steps 15 --quality-target 0.9 "Quantum computing applications"

# Use specific LLM provider
python cli.py --provider google "Blockchain in supply chain"

# Preview results immediately
python cli.py --preview "Gene therapy breakthroughs"

# Quiet mode (minimal output)
python cli.py --quiet "Space exploration technologies"
```

### Python API

```python
import asyncio
from autonomous_research_swarm import main as research_main

# Basic research
result = asyncio.run(research_main("AI in autonomous vehicles"))
print(f"Research completed: {result['run_dir']}")

# With custom configuration
config_overrides = {
    'max_steps': 10,
    'quality_target': 0.85,
    'parallel_researchers': 6
}
result = asyncio.run(research_main(
    "Renewable energy storage solutions", 
    config_overrides
))
```

## 🔧 Configuration

### Main Configuration (`config.yaml`)

```yaml
# Core research parameters
max_steps: 12
converge_loops: 3
coverage_target: 0.88
quality_target: 0.85
parallel_researchers: 4

# LLM settings
llm:
  provider: perplexity  # perplexity | google | moonshot
  model: sonar-large-online
  temperature: 0.2

# Safety controls
safety:
  allow_domains: [".edu", ".gov", ".org"]
  deny_domains: ["spam.site"]
  obey_robots: true

# Resource limits
budgets:
  max_minutes: 25
  max_tokens: 200000
  max_fetches: 150
```

### Environment Variables (Optional)

While API keys are pre-configured, you can override them:

```bash
export PERPLEXITY_API_KEY="your-key-here"
export GOOGLE_AI_API_KEY="your-key-here"
export HUGGINGFACE_TOKEN="your-token-here"
```

## 📊 Output Structure

Each research run creates a timestamped directory with:

```
runs/2024-01-15T14-30-45-abc123/
├── report.md           # Final research report
├── evidence.jsonl      # All evidence with citations
├── plan.json          # Research plan and task breakdown
├── events.jsonl       # Complete audit log
└── checkpoints/       # Intermediate snapshots
```

### Sample Report Structure

```markdown
# Research Report: AI in Healthcare Workflows

## Executive Summary
[High-level findings and conclusions]

## Key Findings
1. **Diagnostic Accuracy**: AI systems show 95% accuracy in radiology (Source: [1])
2. **Workflow Efficiency**: 40% reduction in processing time (Source: [2])

## Current State Analysis
[Detailed analysis with citations]

## Challenges and Limitations
[Identified issues and constraints]

## Future Directions
[Emerging trends and predictions]

## Conclusion
[Summary and implications]

---
*Generated by Jukeyman Research Swarm - Rick Jefferson Solutions*
*Sources: 23 | Quality Score: 0.89 | Coverage: 0.92*
```

## 🎛️ Advanced Features

### Multi-Agent Coordination

```python
# The system automatically coordinates multiple agents:
# 1. Planner creates research subtasks
# 2. Researchers execute searches in parallel
# 3. Librarian deduplicates and ranks sources
# 4. Analyst synthesizes findings
# 5. Critic evaluates quality and identifies gaps
# 6. Orchestrator decides whether to continue or stop
```

### Quality Convergence

```python
# Research continues until:
# - Quality score ≥ quality_target (default: 0.85)
# - Coverage score ≥ coverage_target (default: 0.88)
# - No improvement for converge_loops iterations (default: 3)
# - Budget limits reached (time/tokens/requests)
```

### Source Authority Ranking

```python
# Sources are ranked by authority:
authority_weights = {
    'gov': 1.0,    # Government sources
    'edu': 0.95,   # Educational institutions
    'org': 0.85,   # Organizations
    'com': 0.65    # Commercial sites
}
```

## 🔍 Research Topics Examples

The system excels at researching:

### 🏥 **Healthcare & Medicine**
- "Impact of Large Language Models on radiology workflows (2020-2025)"
- "Gene therapy breakthroughs in treating rare diseases"
- "AI-assisted drug discovery and development timelines"

### 🌍 **Climate & Environment**
- "Climate change effects on global food security and agriculture"
- "Renewable energy storage solutions and grid integration"
- "Carbon capture technologies and scalability challenges"

### 💻 **Technology & AI**
- "Quantum computing applications in cryptography and security"
- "Artificial intelligence in autonomous vehicle safety systems"
- "Blockchain technology adoption in supply chain management"

### 🧬 **Science & Research**
- "CRISPR gene editing applications and ethical considerations"
- "Space exploration technologies and Mars colonization prospects"
- "Fusion energy progress and commercial viability timeline"

### 🏛️ **Social & Economic**
- "Social media impact on mental health in adolescents"
- "Remote work effects on productivity and corporate culture"
- "Cryptocurrency regulation and financial system integration"

## 🛠️ Development

### Project Structure

```
rick-jefferson-solutions/
├── autonomous_research_swarm.py  # Main research engine
├── cli.py                       # Command-line interface
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
├── README.md                   # This file
└── runs/                       # Research outputs
    └── [timestamp-folders]/
```

### Key Components

- **`autonomous_research_swarm.py`**: Core research engine with all agents
- **`cli.py`**: Rich command-line interface with progress tracking
- **`config.yaml`**: Comprehensive configuration options
- **API Integration**: Pre-configured with multiple AI providers

### Extending the System

```python
# Add new LLM provider
async def custom_llm(prompt: str, system: str = ""):
    # Your implementation here
    return response

# Add to call_llm function
async def call_llm(prompt, system="", provider="custom"):
    if provider == "custom":
        return await custom_llm(prompt, system)
    # ... existing providers
```

## 📈 Performance & Scaling

### Typical Performance
- **Research Time**: 5-15 minutes per topic
- **Sources Analyzed**: 20-50 per research session
- **Quality Score**: Usually achieves 0.85+ on comprehensive topics
- **Parallel Processing**: 3-6 researchers working simultaneously

### Optimization Tips

```yaml
# For faster research (lower quality)
max_steps: 8
parallel_researchers: 6
quality_target: 0.75

# For comprehensive research (higher quality)
max_steps: 15
parallel_researchers: 3
quality_target: 0.90
coverage_target: 0.95
```

## 🔒 Security & Privacy

- **API Key Management**: Secure handling of authentication tokens
- **Rate Limiting**: Respects service provider limits
- **Content Filtering**: Blocks malicious and inappropriate content
- **Privacy**: No personal data collection or storage
- **Compliance**: Respects robots.txt and crawling policies

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional LLM providers (OpenAI, Anthropic, etc.)
- Enhanced content extraction (PDF, academic papers)
- Real-time collaboration features
- Web interface development
- Performance optimizations

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Perplexity AI**: For powerful search and reasoning capabilities
- **Google AI**: For Gemini model access
- **Hugging Face**: For open-source embeddings and models
- **Research Community**: For inspiration and best practices

---

**Ready to start researching?** 🚀

```bash
python cli.py --interactive
```

*Let the AI swarm handle the heavy lifting while you focus on insights and decisions.*