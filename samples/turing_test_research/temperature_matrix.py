#!/usr/bin/env python3
"""
Temperature Matrix Testing for AutoGen Turing Test Extension

This script reproduces the systematic temperature matrix testing approach
from the original Turing Test implementation, now using proper AutoGen architecture.

Based on the original research by Jacob Choi and Mark Encarnación.
"""

import asyncio
import logging
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import AutoGen components
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.teams.turing_bench import TuringTestTeam

class TemperatureMatrixExperiment:
    """
    Conducts systematic temperature matrix experiments for Turing Test research.
    """
    
    def __init__(
        self,
        model_client: Any,
        experiment_name: str = None,
        results_dir: str = "temperature_matrix_results"
    ):
        self.model_client = model_client
        self.experiment_name = experiment_name or f"temp_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize Turing Test team
        self.turing_team = TuringTestTeam(
            model_client=model_client,
            results_dir=str(self.results_dir / self.experiment_name)
        )
        
        logger.info(f"Initialized experiment: {self.experiment_name}")
    
    async def run_full_matrix(
        self,
        temperature_range: tuple = (0.1, 0.9, 0.1),
        interrogator_types: List[str] = None,
        tests_per_temperature: int = 10,
        include_human_proxy: bool = True,
        max_rounds: int = 10
    ) -> Dict[str, Any]:
        """
        Run the complete temperature matrix experiment.
        
        Args:
            temperature_range: (min, max, step) for temperatures
            interrogator_types: List of interrogator types to test
            tests_per_temperature: Number of tests per temperature
            include_human_proxy: Whether to include human proxy tests
            max_rounds: Maximum conversation rounds
            
        Returns:
            Complete experiment results
        """
        
        if interrogator_types is None:
            interrogator_types = ["investigative", "casual", "philosophical"]
        
        # Update team configuration
        self.turing_team.max_rounds = max_rounds
        
        logger.info(f"🚀 Starting Full Temperature Matrix Experiment")
        logger.info(f"📊 Configuration:")
        logger.info(f"   Experiment: {self.experiment_name}")
        logger.info(f"   Temperature range: {temperature_range}")
        logger.info(f"   Interrogator types: {interrogator_types}")
        logger.info(f"   Tests per temperature: {tests_per_temperature}")
        logger.info(f"   Include human proxy: {include_human_proxy}")
        logger.info(f"   Max rounds per test: {max_rounds}")
        
        # Calculate total tests
        min_temp, max_temp, step = temperature_range
        num_temperatures = len([t for t in self._generate_temperatures(temperature_range)])
        total_tests = len(interrogator_types) * num_temperatures * tests_per_temperature
        if include_human_proxy:
            total_tests *= 2
        
        logger.info(f"🎯 Total tests to run: {total_tests}")
        estimated_time = total_tests * 2  # Rough estimate: 2 minutes per test
        logger.info(f"⏱️  Estimated duration: {estimated_time // 60}h {estimated_time % 60}m")
        
        # Run the experiment
        start_time = datetime.now()
        
        try:
            results = await self.turing_team.run_temperature_matrix(
                interrogator_types=interrogator_types,
                temperature_range=temperature_range,
                tests_per_temperature=tests_per_temperature,
                include_human_proxy=include_human_proxy,
                experiment_name=self.experiment_name
            )
            
            end_time = datetime.now()
            actual_duration = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Experiment completed!")
            logger.info(f"⏱️  Actual duration: {actual_duration // 3600:.0f}h {(actual_duration % 3600) // 60:.0f}m")
            logger.info(f"📈 Overall accuracy: {results['analysis']['overall_accuracy']:.1%}")
            
            # Generate visualizations
            await self._generate_visualizations(results)
            
            # Generate research report
            await self._generate_research_report(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            raise
    
    async def run_quick_matrix(
        self,
        temperature_range: tuple = (0.1, 0.9, 0.4),  # Fewer temperatures
        tests_per_temperature: int = 3,
        max_rounds: int = 5
    ) -> Dict[str, Any]:
        """
        Run a quick version for testing/development.
        """
        
        logger.info("🏃‍♂️ Running Quick Temperature Matrix (for testing)")
        
        return await self.run_full_matrix(
            temperature_range=temperature_range,
            interrogator_types=["investigative"],  # Just one interrogator type
            tests_per_temperature=tests_per_temperature,
            include_human_proxy=False,  # AI only for speed
            max_rounds=max_rounds
        )
    
    async def run_replication_study(
        self,
        original_results_file: str = None
    ) -> Dict[str, Any]:
        """
        Run a replication of previous Turing Test results.
        """
        
        logger.info("🔄 Running Replication Study")
        logger.info("   This replicates the original 'vibe coded' research with proper methodology")
        
        # Use same parameters as original study
        return await self.run_full_matrix(
            temperature_range=(0.1, 0.9, 0.1),  # 0.1 to 0.9 in 0.1 steps
            interrogator_types=["investigative", "casual", "philosophical"],
            tests_per_temperature=10,  # Match original: 100 conversations per temperature
            include_human_proxy=True,
            max_rounds=10
        )
    
    def _generate_temperatures(self, temperature_range: tuple) -> List[float]:
        """Generate list of temperatures from range."""
        min_temp, max_temp, step = temperature_range
        temperatures = []
        temp = min_temp
        while temp <= max_temp + 1e-9:  # Small epsilon for floating point comparison
            temperatures.append(round(temp, 1))
            temp += step
        return temperatures
    
    async def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate visualizations of experiment results."""
        
        logger.info("📊 Generating visualizations...")
        
        # Create DataFrame from results
        individual_results = results["individual_results"]
        
        df_data = []
        for result in individual_results:
            df_data.append({
                "temperature": result["configuration"]["subject_temperature"],
                "interrogator_type": result["configuration"]["interrogator_type"],
                "subject_type": result["configuration"]["subject_type"],
                "correct_identification": result["evaluation"]["correct_identification"],
                "confidence_score": result["evaluation"]["confidence_score"],
                "authenticity_score": result["evaluation"]["authenticity_score"],
                "conversation_quality": result["evaluation"]["conversation_quality"],
                "total_rounds": result["evaluation"]["total_rounds"],
                "test_id": result["test_id"]
            })
        
        df = pd.DataFrame(df_data)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Temperature Matrix Analysis: {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # 1. Accuracy by Temperature
        temp_accuracy = df.groupby('temperature')['correct_identification'].mean()
        axes[0, 0].plot(temp_accuracy.index, temp_accuracy.values, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Identification Accuracy by Temperature')
        axes[0, 0].set_xlabel('Temperature')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Add accuracy values as text
        for temp, acc in temp_accuracy.items():
            axes[0, 0].text(temp, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom')
        
        # 2. Heatmap: Temperature vs Interrogator Type
        if len(df['interrogator_type'].unique()) > 1:
            heatmap_data = df.groupby(['temperature', 'interrogator_type'])['correct_identification'].mean().unstack()
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0, 1], cbar_kws={'label': 'Accuracy'})
            axes[0, 1].set_title('Accuracy Heatmap: Temperature × Interrogator')
        else:
            # If only one interrogator type, show confidence scores
            temp_confidence = df.groupby('temperature')['confidence_score'].mean()
            axes[0, 1].plot(temp_confidence.index, temp_confidence.values, 's-', color='orange', linewidth=2, markersize=8)
            axes[0, 1].set_title('Confidence Score by Temperature')
            axes[0, 1].set_xlabel('Temperature')
            axes[0, 1].set_ylabel('Confidence Score')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Score Distribution by Temperature
        score_cols = ['confidence_score', 'authenticity_score', 'conversation_quality']
        temp_scores = df.groupby('temperature')[score_cols].mean()
        
        for i, col in enumerate(score_cols):
            axes[1, 0].plot(temp_scores.index, temp_scores[col], 'o-', label=col.replace('_', ' ').title(), linewidth=2, markersize=6)
        
        axes[1, 0].set_title('Performance Scores by Temperature')
        axes[1, 0].set_xlabel('Temperature')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # 4. Subject Type Comparison (if both AI and human proxy tested)
        if len(df['subject_type'].unique()) > 1:
            subject_accuracy = df.groupby(['temperature', 'subject_type'])['correct_identification'].mean().unstack()
            for subject_type in subject_accuracy.columns:
                axes[1, 1].plot(subject_accuracy.index, subject_accuracy[subject_type], 'o-', 
                              label=f'{subject_type.title()} Subject', linewidth=2, markersize=6)
            axes[1, 1].set_title('Accuracy by Subject Type')
            axes[1, 1].set_xlabel('Temperature')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Show conversation rounds distribution
            round_dist = df.groupby('temperature')['total_rounds'].mean()
            axes[1, 1].bar(round_dist.index, round_dist.values, alpha=0.7, color='skyblue')
            axes[1, 1].set_title('Average Conversation Rounds by Temperature')
            axes[1, 1].set_xlabel('Temperature')
            axes[1, 1].set_ylabel('Average Rounds')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.results_dir / self.experiment_name / f"{self.experiment_name}_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualization saved: {viz_path}")
        
        plt.show()
        
        # Save the DataFrame for further analysis
        csv_path = self.results_dir / self.experiment_name / f"{self.experiment_name}_data.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"📁 Data saved: {csv_path}")
    
    async def _generate_research_report(self, results: Dict[str, Any]):
        """Generate a research report summary."""
        
        logger.info("📝 Generating research report...")
        
        analysis = results["analysis"]
        config = results["configuration"]
        
        report = f"""
# Temperature Matrix Experiment Report
**Experiment ID:** {self.experiment_name}
**Date:** {results['timestamp']}

## Experimental Configuration
- **Temperature Range:** {config['temperature_range'][0]} to {config['temperature_range'][1]} (step: {config['temperature_range'][2]})
- **Interrogator Types:** {', '.join(config['interrogator_types'])}
- **Tests per Temperature:** {config['tests_per_temperature']}
- **Include Human Proxy:** {config['include_human_proxy']}
- **Total Tests:** {config['total_tests']}

## Key Findings

### Overall Performance
- **Overall Accuracy:** {analysis['overall_accuracy']:.1%} ({analysis['correct_identifications']}/{analysis['total_tests']} tests)
- **Best Temperature:** {analysis['best_temperature']} (accuracy: {analysis['performance_by_temperature'][analysis['best_temperature']]['accuracy']:.1%})
- **Best Interrogator:** {analysis['best_interrogator']} (accuracy: {analysis['performance_by_interrogator'][analysis['best_interrogator']]['accuracy']:.1%})

### Performance by Temperature
"""
        
        # Add temperature performance details
        for temp, perf in sorted(analysis['performance_by_temperature'].items()):
            report += f"- **T={temp}:** {perf['accuracy']:.1%} accuracy ({perf['correct']}/{perf['total']} tests)\n"
        
        report += f"""

### Performance by Interrogator Type
"""
        
        # Add interrogator performance details
        for interr, perf in analysis['performance_by_interrogator'].items():
            report += f"- **{interr.title()}:** {perf['accuracy']:.1%} accuracy ({perf['correct']}/{perf['total']} tests)\n"
        
        report += f"""

### Average Scores
- **Confidence:** {analysis['average_scores']['confidence']:.2f}
- **Authenticity:** {analysis['average_scores']['authenticity']:.2f}
- **Quality:** {analysis['average_scores']['quality']:.2f}

## Research Implications

### Temperature Analysis
The optimal temperature of {analysis['best_temperature']} suggests {'higher' if analysis['best_temperature'] > 0.5 else 'lower'} randomness {'improves' if analysis['performance_by_temperature'][analysis['best_temperature']]['accuracy'] > 0.5 else 'reduces'} human-like behavior in AI responses.

### Interrogator Effectiveness
The {analysis['best_interrogator']} interrogator type proved most effective at distinguishing human from AI responses, indicating that {'systematic questioning' if analysis['best_interrogator'] == 'investigative' else 'conversational approach' if analysis['best_interrogator'] == 'casual' else 'abstract reasoning tasks'} may be key discrimination factors.

## Files Generated
- Full results: `{self.experiment_name}_results.json`
- Data summary: `{self.experiment_name}_summary.csv`
- Data analysis: `{self.experiment_name}_data.csv`
- Visualization: `{self.experiment_name}_analysis.png`
- This report: `{self.experiment_name}_report.md`

---
*Generated by AutoGen Turing Test Extension*
*Original research by Jacob Choi (Colby College) and Mark Encarnación (Microsoft Research)*
"""
        
        # Save report
        report_path = self.results_dir / self.experiment_name / f"{self.experiment_name}_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"📝 Research report saved: {report_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("🏆 EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Overall Accuracy: {analysis['overall_accuracy']:.1%}")
        print(f"Best Temperature: {analysis['best_temperature']}")
        print(f"Best Interrogator: {analysis['best_interrogator']}")
        print(f"Total Tests: {analysis['total_tests']}")
        print("="*60)

async def main():
    """Main function for running temperature matrix experiments."""
    
    parser = argparse.ArgumentParser(description="AutoGen Turing Test Temperature Matrix Experiment")
    parser.add_argument("--mode", choices=["quick", "full", "replication"], default="quick",
                        help="Experiment mode (default: quick)")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--tests-per-temp", type=int, default=10, help="Tests per temperature")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max conversation rounds")
    parser.add_argument("--temp-min", type=float, default=0.1, help="Minimum temperature")
    parser.add_argument("--temp-max", type=float, default=0.9, help="Maximum temperature")
    parser.add_argument("--temp-step", type=float, default=0.1, help="Temperature step size")
    
    args = parser.parse_args()
    
    # Check Azure configuration
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GPT4", "gpt-4")
    
    if not azure_endpoint or not azure_key:
        print("❌ Azure OpenAI configuration missing!")
        print("Please set these environment variables:")
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
    
    # Initialize experiment
    experiment = TemperatureMatrixExperiment(
        model_client=model_client,
        experiment_name=args.name
    )
    
    try:
        # Run experiment based on mode
        if args.mode == "quick":
            print("🏃‍♂️ Running Quick Experiment (for testing)")
            results = await experiment.run_quick_matrix(
                temperature_range=(args.temp_min, args.temp_max, args.temp_step),
                tests_per_temperature=min(args.tests_per_temp, 3),  # Cap at 3 for quick mode
                max_rounds=min(args.max_rounds, 5)  # Cap at 5 for quick mode
            )
            
        elif args.mode == "full":
            print("🚀 Running Full Temperature Matrix Experiment")
            results = await experiment.run_full_matrix(
                temperature_range=(args.temp_min, args.temp_max, args.temp_step),
                tests_per_temperature=args.tests_per_temp,
                max_rounds=args.max_rounds
            )
            
        elif args.mode == "replication":
            print("🔄 Running Replication Study")
            print("   This replicates the original research methodology")
            results = await experiment.run_replication_study()
        
        print(f"\n🎉 Experiment completed successfully!")
        print(f"📁 Results saved in: {experiment.results_dir / experiment.experiment_name}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        print("   Partial results may be available in the results directory")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        logger.exception("Detailed error:")
        
    finally:
        # Clean up
        await model_client.close()

if __name__ == "__main__":
    print("🤖 AutoGen Turing Test Extension - Temperature Matrix Experiment")
    print("=" * 70)
    print("Original research by Jacob Choi (Colby College) & Mark Encarnación (Microsoft Research)")
    print("Rebuilt with proper software engineering practices using AutoGen v0.4")
    print("=" * 70)
    
    asyncio.run(main())