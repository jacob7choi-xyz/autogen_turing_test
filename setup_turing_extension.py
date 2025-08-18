#!/usr/bin/env python3
"""
Setup Script for AutoGen Turing Test Extension

This script sets up the complete Turing Test extension in your AutoGen repository,
creating all necessary files and directories.

Usage:
    python setup_turing_extension.py
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def create_directory_structure():
    """Create the directory structure for the Turing Test extension."""
    
    print("📁 Creating directory structure...")
    
    # Base paths
    base_dir = Path.cwd()
    autogen_ext_path = base_dir / "python/packages/autogen-ext/src/autogen_ext"
    samples_path = base_dir / "samples/turing_test_research"
    
    # Turing Test agent directories
    turing_agents_path = autogen_ext_path / "agents/turing_test"
    turing_teams_path = autogen_ext_path / "teams"
    
    # Create directories
    directories = [
        turing_agents_path,
        samples_path,
        samples_path / "configs",
        samples_path / "results", 
        samples_path / "notebooks",
        base_dir / "tests/agents/turing_test",
        base_dir / "tests/teams/turing_bench"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    return {
        "turing_agents": turing_agents_path,
        "turing_teams": turing_teams_path,
        "samples": samples_path,
        "base": base_dir
    }

def create_agent_files(paths):
    """Create the core agent files."""
    
    print("🤖 Creating agent files...")
    
    agents_dir = paths["turing_agents"]
    
    # Agent files content
    files = {
        "__init__.py": '''"""
AutoGen Turing Test Extension

This module provides agents and orchestration for conducting Turing Test experiments
within the AutoGen framework. It enables systematic evaluation of AI agents through
human-AI discrimination tasks with temperature matrix testing.

Authors: Jacob Choi (Colby College), Mark Encarnación (Microsoft Research)
"""

from ._interrogator_agent import InterrogatorAgent
from ._subject_agent import SubjectAgent  
from ._evaluator_agent import EvaluatorAgent

__all__ = [
    "InterrogatorAgent",
    "SubjectAgent", 
    "EvaluatorAgent"
]

__version__ = "0.1.0"
''',
        
        "README.md": '''# AutoGen Turing Test Extension

This extension provides a comprehensive framework for conducting Turing Test experiments using the AutoGen multi-agent system.

## Components

### Agents
- **InterrogatorAgent**: Conducts conversations designed to distinguish human from AI
- **SubjectAgent**: Represents test participants (AI or human proxy)
- **EvaluatorAgent**: Analyzes conversations and provides detailed assessments

### Teams
- **TuringTestTeam**: Orchestrates complete experiments with temperature matrix testing

## Features

- Temperature matrix testing (systematic evaluation across temperature ranges)
- Multiple interrogator types (investigative, casual, philosophical, creative)
- Human proxy simulation for baseline comparison
- Comprehensive evaluation metrics and analysis
- Cost tracking and performance monitoring
- Research-quality data collection and visualization

## Usage

```python
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.teams.turing_bench import run_quick_turing_test

# Setup model client
model_client = OpenAIChatCompletionClient(...)

# Run a single test
result = await run_quick_turing_test(
    model_client=model_client,
    interrogator_type="investigative",
    subject_type="ai",
    temperature=0.7
)
```

## Research Background

This implementation is based on Turing Test research conducted by Jacob Choi (Colby College) and Mark Encarnación (Microsoft Research), focusing on systematic evaluation of AI human-likeness through conversational discrimination tasks.
''',
        
        "_interrogator_agent.py": "# Interrogator Agent implementation\n# (Content from previous artifact)",
        "_subject_agent.py": "# Subject Agent implementation\n# (Content from previous artifact)", 
        "_evaluator_agent.py": "# Evaluator Agent implementation\n# (Content from previous artifact)"
    }
    
    for filename, content in files.items():
        file_path = agents_dir / filename
        if not file_path.exists() or filename == "README.md":  # Always update README
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"   ✅ {file_path}")
        else:
            print(f"   ⚠️  {file_path} already exists, skipping")

def create_teams_file(paths):
    """Create the teams orchestration file."""
    
    print("🎯 Creating teams file...")
    
    teams_dir = paths["turing_teams"]
    turing_bench_path = teams_dir / "turing_bench.py"
    
    # Check if file exists and is empty
    if turing_bench_path.exists():
        with open(turing_bench_path, 'r') as f:
            content = f.read().strip()
        
        if content:
            print(f"   ⚠️  {turing_bench_path} already has content, creating backup")
            backup_path = turing_bench_path.with_suffix('.py.backup')
            shutil.copy2(turing_bench_path, backup_path)
            print(f"   📁 Backup saved: {backup_path}")
    
    # Create/update the file
    teams_content = "# Turing Bench Team implementation\n# (Content from previous artifact)\n"
    
    with open(turing_bench_path, 'w') as f:
        f.write(teams_content)
    print(f"   ✅ {turing_bench_path}")

def create_sample_files(paths):
    """Create sample and configuration files."""
    
    print("📄 Creating sample files...")
    
    samples_dir = paths["samples"]
    
    # Configuration files
    config_files = {
        "configs/azure_openai_config.json": {
            "azure_openai": {
                "endpoint": "your-azure-openai-endpoint",
                "api_key": "your-api-key",
                "api_version": "2024-02-01",
                "gpt4_deployment": "gpt-4-deployment-name",
                "gpt35_deployment": "gpt-35-turbo-deployment-name"
            },
            "turing_test": {
                "max_rounds": 10,
                "timeout_seconds": 300,
                "temperature_range": [0.1, 0.9, 0.1],
                "results_dir": "samples/turing_test_research/results"
            }
        },
        
        "configs/test_scenarios.json": {
            "scenarios": [
                {
                    "name": "basic_discrimination",
                    "interrogator_type": "investigative", 
                    "subject_types": ["ai", "human_proxy"],
                    "temperatures": [0.1, 0.5, 0.9],
                    "tests_per_config": 5
                },
                {
                    "name": "conversational_styles",
                    "interrogator_types": ["casual", "investigative", "philosophical"],
                    "subject_type": "ai",
                    "temperature": 0.7,
                    "tests_per_config": 3
                }
            ]
        }
    }
    
    for filename, content in config_files.items():
        file_path = samples_dir / filename
        if not file_path.exists():
            import json
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
            print(f"   ✅ {file_path}")
    
    # Sample Python files
    sample_files = {
        "basic_conversation.py": "# Basic conversation example\n# (Content from previous artifact)",
        "temperature_matrix.py": "# Temperature matrix testing\n# (Content from previous artifact)",
        "azure_config_example.py": "# Azure configuration example\n# (Content from previous artifact)"
    }
    
    for filename, content in sample_files.items():
        file_path = samples_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"   ✅ {file_path}")

def create_env_template(paths):
    """Create environment template file."""
    
    print("🔐 Creating environment template...")
    
    env_template_path = paths["base"] / ".env.template"
    
    env_content = '''# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=your-azure-openai-endpoint-here
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-01

# Model configurations for Turing Test
AZURE_OPENAI_DEPLOYMENT_NAME_GPT4=gpt-4-deployment-name
AZURE_OPENAI_DEPLOYMENT_NAME_GPT35=gpt-35-turbo-deployment-name

# Turing Test Configuration
TURING_TEST_MAX_ROUNDS=10
TURING_TEST_TIMEOUT_SECONDS=300
TURING_TEST_TEMPERATURE_MIN=0.1
TURING_TEST_TEMPERATURE_MAX=0.9
TURING_TEST_TEMPERATURE_STEP=0.1

# Logging and Results
LOG_LEVEL=INFO
RESULTS_DIR=samples/turing_test_research/results
ENABLE_COST_TRACKING=true

# Development Settings
DEBUG_MODE=false
VERBOSE_LOGGING=false
'''
    
    if not env_template_path.exists():
        with open(env_template_path, 'w') as f:
            f.write(env_content)
        print(f"   ✅ {env_template_path}")
    else:
        print(f"   ⚠️  {env_template_path} already exists")

def create_test_files(paths):
    """Create test files."""
    
    print("🧪 Creating test files...")
    
    base_dir = paths["base"]
    
    test_files = {
        "tests/agents/turing_test/test_interrogator.py": "# Interrogator agent tests",
        "tests/agents/turing_test/test_subject.py": "# Subject agent tests",
        "tests/agents/turing_test/test_evaluator.py": "# Evaluator agent tests",
        "tests/teams/turing_bench/test_turing_team.py": "# Turing test team tests"
    }
    
    for filename, content in test_files.items():
        file_path = base_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                f.write(content + "\n")
            print(f"   ✅ {file_path}")

def create_readme_updates(paths):
    """Create README file for samples."""
    
    print("📖 Creating documentation...")
    
    samples_readme = paths["samples"] / "README.md"
    
    readme_content = '''# AutoGen Turing Test Research

This directory contains research tools and examples for conducting Turing Test experiments with AutoGen.

## Quick Start

1. **Setup Environment**
   ```bash
   # Copy environment template
   cp ../../.env.template ../../.env
   
   # Edit .env with your Azure OpenAI credentials
   nano ../../.env
   ```

2. **Run Basic Test**
   ```bash
   python basic_conversation.py
   ```

3. **Run Temperature Matrix**
   ```bash
   # Quick test (3 temperatures, 3 tests each)
   python temperature_matrix.py --mode quick
   
   # Full experiment (research-grade)
   python temperature_matrix.py --mode full
   ```

## Files

### Core Examples
- `basic_conversation.py` - Single Turing Test demonstration
- `temperature_matrix.py` - Systematic temperature testing
- `azure_config_example.py` - Azure OpenAI setup and testing

### Configuration
- `configs/azure_openai_config.json` - Azure OpenAI configuration
- `configs/test_scenarios.json` - Predefined test scenarios

### Results
- `results/` - Experiment results and analysis
- `notebooks/` - Jupyter notebooks for data analysis

## Research Background

This implementation rebuilds the original Turing Test research by Jacob Choi (Colby College) and Mark Encarnación (Microsoft Research) using proper software engineering practices and the AutoGen v0.4 framework.

### Key Improvements
- Modular agent architecture following AutoGen patterns
- Comprehensive evaluation metrics
- Temperature matrix testing capabilities
- Cost tracking and monitoring
- Research-quality data collection
- Reproducible experiment methodology

## Usage Examples

### Single Test
```python
from autogen_ext.teams.turing_bench import run_quick_turing_test
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(...)
result = await run_quick_turing_test(
    model_client=model_client,
    interrogator_type="investigative",
    subject_type="ai",
    temperature=0.7
)
```

### Full Experiment
```python
from autogen_ext.teams.turing_bench import TuringTestTeam

team = TuringTestTeam(model_client)
results = await team.run_temperature_matrix(
    temperature_range=(0.1, 0.9, 0.1),
    tests_per_temperature=10
)
```
'''
    
    with open(samples_readme, 'w') as f:
        f.write(readme_content)
    print(f"   ✅ {samples_readme}")

def main():
    """Main setup function."""
    
    print("🚀 AutoGen Turing Test Extension Setup")
    print("=" * 50)
    print("Setting up the complete Turing Test extension...")
    print()
    
    try:
        # Create directory structure
        paths = create_directory_structure()
        
        # Create all files
        create_agent_files(paths)
        create_teams_file(paths) 
        create_sample_files(paths)
        create_env_template(paths)
        create_test_files(paths)
        create_readme_updates(paths)
        
        print("\n🎉 Setup completed successfully!")
        print("=" * 50)
        
        print("\n📋 Next Steps:")
        print("1. Copy .env.template to .env and configure Azure OpenAI credentials")
        print("2. Navigate to samples/turing_test_research/")
        print("3. Run: python basic_conversation.py")
        print("4. For research: python temperature_matrix.py --mode quick")
        
        print("\n📁 Files Created:")
        print("   🤖 Agent implementations in python/packages/autogen-ext/src/autogen_ext/agents/turing_test/")
        print("   🎯 Team orchestration in python/packages/autogen-ext/src/autogen_ext/teams/")
        print("   📄 Sample scripts in samples/turing_test_research/")
        print("   🧪 Test files in tests/")
        
        print("\n💡 Important Notes:")
        print("   - This setup creates file stubs - you'll need to copy the full implementations")
        print("   - Configure Azure OpenAI credentials before running")
        print("   - Start with quick mode for testing, full mode for research")
        
        print("\n✨ Ready to conduct Turing Test research with AutoGen!")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        raise

if __name__ == "__main__":
    main()