#!/usr/bin/env python3
"""
Basic Turing Test Conversation Example

This script demonstrates how to run a single Turing Test conversation
using the AutoGen Turing Test Extension.

Usage:
    python basic_conversation.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import AutoGen components
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.turing_test import InterrogatorAgent, SubjectAgent, EvaluatorAgent
from autogen_ext.teams.turing_bench import run_quick_turing_test

async def demo_single_test():
    """Demonstrate a single Turing Test conversation."""
    
    print("🤖 AutoGen Turing Test Extension Demo")
    print("=" * 50)
    
    # Configure Azure OpenAI client
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GPT4", "gpt-4")
    
    if not azure_endpoint or not azure_key:
        print("❌ Please configure Azure OpenAI credentials in .env file")
        print("Required variables:")
        print("  - AZURE_OPENAI_ENDPOINT")
        print("  - AZURE_OPENAI_API_KEY")
        print("  - AZURE_OPENAI_DEPLOYMENT_NAME_GPT4 (optional)")
        return
    
    # Create model client
    try:
        model_client = OpenAIChatCompletionClient(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            model=deployment_name,
            api_version="2024-02-01"
        )
        print(f"✅ Connected to Azure OpenAI: {deployment_name}")
    except Exception as e:
        print(f"❌ Failed to connect to Azure OpenAI: {e}")
        return
    
    # Run a quick test
    print("\n🎯 Running Turing Test...")
    print("Configuration:")
    print("  - Interrogator: Investigative")
    print("  - Subject: AI (Temperature 0.7)")
    print("  - Max Rounds: 5")
    
    try:
        result = await run_quick_turing_test(
            model_client=model_client,
            interrogator_type="investigative",
            subject_type="ai", 
            temperature=0.7,
            max_rounds=5
        )
        
        # Display results
        print("\n📊 Test Results:")
        print("=" * 30)
        
        eval_data = result["evaluation"]
        config_data = result["configuration"]
        
        print(f"Test ID: {result['test_id']}")
        print(f"Interrogator Verdict: {eval_data['interrogator_verdict']}")
        print(f"Actual Subject Type: {eval_data['actual_type']}")
        print(f"Correct Identification: {'✅ YES' if eval_data['correct_identification'] else '❌ NO'}")
        print(f"Confidence Score: {eval_data['confidence_score']:.2f}")
        print(f"Authenticity Score: {eval_data['authenticity_score']:.2f}")
        print(f"Conversation Quality: {eval_data['conversation_quality']:.2f}")
        print(f"Total Rounds: {eval_data['total_rounds']}")
        print(f"Duration: {eval_data['conversation_duration']:.1f} seconds")
        
        if eval_data.get('key_indicators'):
            print(f"\nKey Indicators:")
            for indicator in eval_data['key_indicators']:
                print(f"  • {indicator}")
        
        if eval_data.get('evaluator_notes'):
            print(f"\nEvaluator Notes:")
            print(f"  {eval_data['evaluator_notes'][:200]}...")
        
        print(f"\n💾 Results saved to: {result.get('saved_to', 'quick_test_results/')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Detailed error:")
    
    finally:
        # Clean up
        await model_client.close()

async def demo_multiple_temperatures():
    """Demonstrate testing multiple temperatures."""
    
    print("\n🌡️  Temperature Testing Demo")
    print("=" * 50)
    
    # Configure client (reuse from above)
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY") 
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GPT4", "gpt-4")
    
    if not azure_endpoint or not azure_key:
        print("❌ Azure OpenAI not configured")
        return
    
    model_client = OpenAIChatCompletionClient(
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        model=deployment_name,
        api_version="2024-02-01"
    )
    
    temperatures = [0.1, 0.5, 0.9]
    results = []
    
    print(f"Testing temperatures: {temperatures}")
    print("Each test: Investigative interrogator vs AI subject")
    
    try:
        for temp in temperatures:
            print(f"\n🌡️  Testing temperature {temp}...")
            
            result = await run_quick_turing_test(
                model_client=model_client,
                interrogator_type="investigative",
                subject_type="ai",
                temperature=temp,
                max_rounds=3  # Shorter for demo
            )
            
            results.append({
                "temperature": temp,
                "correct": result["evaluation"]["correct_identification"],
                "confidence": result["evaluation"]["confidence_score"],
                "authenticity": result["evaluation"]["authenticity_score"]
            })
            
            status = "✅ CORRECT" if result["evaluation"]["correct_identification"] else "❌ INCORRECT"
            print(f"  Result: {status} (Confidence: {result['evaluation']['confidence_score']:.2f})")
        
        # Summary
        print("\n📈 Temperature Analysis:")
        print("Temp | Correct | Confidence | Authenticity")
        print("-" * 40)
        for r in results:
            correct_icon = "✅" if r["correct"] else "❌"
            print(f"{r['temperature']:4.1f} | {correct_icon:7} | {r['confidence']:10.2f} | {r['authenticity']:12.2f}")
        
        accuracy = sum(1 for r in results if r["correct"]) / len(results)
        print(f"\nOverall Accuracy: {accuracy:.1%}")
        
    except Exception as e:
        print(f"❌ Temperature testing failed: {e}")
        logger.exception("Detailed error:")
    
    finally:
        await model_client.close()

async def demo_different_interrogators():
    """Demonstrate different interrogator types."""
    
    print("\n🔍 Interrogator Types Demo")
    print("=" * 50)
    
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GPT4", "gpt-4")
    
    if not azure_endpoint or not azure_key:
        print("❌ Azure OpenAI not configured")
        return
    
    model_client = OpenAIChatCompletionClient(
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        model=deployment_name,
        api_version="2024-02-01"
    )
    
    interrogator_types = ["casual", "investigative", "philosophical", "creative"]
    results = []
    
    print(f"Testing interrogator types: {interrogator_types}")
    print("Each test: Various interrogators vs AI subject (T=0.7)")
    
    try:
        for interr_type in interrogator_types:
            print(f"\n🔍 Testing {interr_type} interrogator...")
            
            result = await run_quick_turing_test(
                model_client=model_client,
                interrogator_type=interr_type,
                subject_type="ai",
                temperature=0.7,
                max_rounds=3
            )
            
            results.append({
                "interrogator": interr_type,
                "correct": result["evaluation"]["correct_identification"],
                "confidence": result["evaluation"]["confidence_score"],
                "quality": result["evaluation"]["conversation_quality"]
            })
            
            status = "✅ CORRECT" if result["evaluation"]["correct_identification"] else "❌ INCORRECT"
            print(f"  Result: {status} (Quality: {result['evaluation']['conversation_quality']:.2f})")
        
        # Summary
        print("\n📊 Interrogator Analysis:")
        print("Type         | Correct | Confidence | Quality")
        print("-" * 45)
        for r in results:
            correct_icon = "✅" if r["correct"] else "❌"
            print(f"{r['interrogator']:12} | {correct_icon:7} | {r['confidence']:10.2f} | {r['quality']:7.2f}")
        
        best_interrogator = max(results, key=lambda x: x["confidence"])
        print(f"\nBest Interrogator: {best_interrogator['interrogator']} (Confidence: {best_interrogator['confidence']:.2f})")
        
    except Exception as e:
        print(f"❌ Interrogator testing failed: {e}")
        logger.exception("Detailed error:")
    
    finally:
        await model_client.close()

def print_menu():
    """Print the demo menu."""
    print("\n🎯 AutoGen Turing Test Extension - Demo Menu")
    print("=" * 50)
    print("1. Single Turing Test")
    print("2. Temperature Testing")
    print("3. Interrogator Types Testing")
    print("4. Exit")
    print()

async def main():
    """Main demo function."""
    
    print("🤖 Welcome to AutoGen Turing Test Extension!")
    print("This demo showcases the key features of the extension.")
    
    while True:
        print_menu()
        choice = input("Select an option (1-4): ").strip()
        
        if choice == "1":
            await demo_single_test()
        elif choice == "2":
            await demo_multiple_temperatures()
        elif choice == "3":
            await demo_different_interrogators()
        elif choice == "4":
            print("👋 Thanks for trying AutoGen Turing Test Extension!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")
        
        if choice in ["1", "2", "3"]:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    asyncio.run(main())