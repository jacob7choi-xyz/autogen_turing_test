"""
Turing Bench - Team orchestration for AutoGen Turing Test Extension

This module provides high-level orchestration for conducting systematic
Turing Test experiments with temperature matrix testing and evaluation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import os
from pathlib import Path
import csv

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import MaxMessageCondition
from autogen_core.models import ChatCompletionClient

# Import our Turing Test agents (assuming they're in the same autogen_ext package)
from autogen_ext.agents.turing_test import InterrogatorAgent, SubjectAgent, EvaluatorAgent

logger = logging.getLogger(__name__)

class TuringTestTeam:
    """
    Orchestrates complete Turing Test experiments with multiple participants,
    temperature testing, and comprehensive evaluation.
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        evaluator_client: Optional[ChatCompletionClient] = None,
        max_rounds: int = 10,
        results_dir: str = "turing_test_results",
        **kwargs
    ):
        """
        Initialize the Turing Test team.
        
        Args:
            model_client: Primary model client for agents
            evaluator_client: Separate model client for evaluator (optional)
            max_rounds: Maximum conversation rounds per test
            results_dir: Directory to save results
            **kwargs: Additional configuration
        """
        
        self.model_client = model_client
        self.evaluator_client = evaluator_client or model_client
        self.max_rounds = max_rounds
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize core components
        self.evaluator = EvaluatorAgent(
            name="TuringEvaluator",
            model_client=self.evaluator_client
        )
        
        self.experiment_results = []
        self.current_experiment_id = None
        
    async def run_single_test(
        self,
        interrogator_type: str = "investigative",
        subject_type: str = "ai",
        subject_temperature: float = 0.7,
        subject_personality: Optional[str] = None,
        test_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a single Turing Test conversation.
        
        Args:
            interrogator_type: Type of interrogator ("investigative", "casual", "philosophical", "creative")
            subject_type: "ai" or "human_proxy"
            subject_temperature: Temperature setting for subject
            subject_personality: Personality type for human proxy subjects
            test_id: Optional test identifier
            
        Returns:
            Dictionary containing test results and analysis
        """
        
        if test_id is None:
            test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        logger.info(f"Starting Turing Test {test_id}: {interrogator_type} vs {subject_type} (T={subject_temperature})")
        
        # Create agents
        interrogator = self._create_interrogator(interrogator_type)
        subject = self._create_subject(subject_type, subject_temperature, subject_personality)
        
        # Run conversation
        conversation_result = await self._run_conversation(interrogator, subject, test_id)
        
        # Evaluate results
        evaluation = await self.evaluator.evaluate_conversation(
            interrogator_data=interrogator.get_conversation_data(),
            subject_data=subject.get_performance_metrics(),
            actual_subject_type=subject_type,
            test_id=test_id
        )
        
        # Compile results
        test_result = {
            "test_id": test_id,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "interrogator_type": interrogator_type,
                "subject_type": subject_type,
                "subject_temperature": subject_temperature,
                "subject_personality": subject_personality,
                "max_rounds": self.max_rounds
            },
            "conversation_data": conversation_result,
            "evaluation": evaluation.__dict__,
            "performance_metrics": {
                "correct_identification": evaluation.correct_identification,
                "confidence_score": evaluation.confidence_score,
                "authenticity_score": evaluation.authenticity_score,
                "conversation_quality": evaluation.conversation_quality
            }
        }
        
        # Save results
        await self._save_test_result(test_result)
        self.experiment_results.append(test_result)
        
        logger.info(f"Completed test {test_id}: {'✅ CORRECT' if evaluation.correct_identification else '❌ INCORRECT'}")
        
        return test_result
    
    async def run_temperature_matrix(
        self,
        interrogator_types: List[str] = None,
        temperature_range: Tuple[float, float, float] = (0.1, 0.9, 0.1),
        tests_per_temperature: int = 5,
        include_human_proxy: bool = True,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive temperature matrix testing.
        
        Args:
            interrogator_types: List of interrogator types to test
            temperature_range: (min, max, step) for temperature testing
            tests_per_temperature: Number of tests per temperature setting
            include_human_proxy: Whether to test human proxy subjects
            experiment_name: Name for this experiment
            
        Returns:
            Comprehensive experiment results
        """
        
        if experiment_name is None:
            experiment_name = f"temp_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_experiment_id = experiment_name
        
        if interrogator_types is None:
            interrogator_types = ["investigative", "casual", "philosophical"]
        
        # Generate temperature list
        min_temp, max_temp, step = temperature_range
        temperatures = []
        temp = min_temp
        while temp <= max_temp:
            temperatures.append(round(temp, 1))
            temp += step
        
        logger.info(f"Starting temperature matrix experiment: {experiment_name}")
        logger.info(f"Interrogators: {interrogator_types}")
        logger.info(f"Temperatures: {temperatures}")
        logger.info(f"Tests per temperature: {tests_per_temperature}")
        
        experiment_results = []
        total_tests = len(interrogator_types) * len(temperatures) * tests_per_temperature
        if include_human_proxy:
            total_tests *= 2  # AI + Human proxy
        
        test_count = 0
        
        for interrogator_type in interrogator_types:
            for temperature in temperatures:
                for test_num in range(tests_per_temperature):
                    # Test AI subject
                    test_count += 1
                    logger.info(f"Progress: {test_count}/{total_tests} - AI subject T={temperature}")
                    
                    ai_result = await self.run_single_test(
                        interrogator_type=interrogator_type,
                        subject_type="ai",
                        subject_temperature=temperature,
                        test_id=f"{experiment_name}_ai_{interrogator_type}_T{temperature}_{test_num}"
                    )
                    experiment_results.append(ai_result)
                    
                    # Test human proxy subject (if enabled)
                    if include_human_proxy:
                        test_count += 1
                        logger.info(f"Progress: {test_count}/{total_tests} - Human proxy T={temperature}")
                        
                        human_result = await self.run_single_test(
                            interrogator_type=interrogator_type,
                            subject_type="human_proxy",
                            subject_temperature=temperature,
                            subject_personality="random",
                            test_id=f"{experiment_name}_human_{interrogator_type}_T{temperature}_{test_num}"
                        )
                        experiment_results.append(human_result)
                    
                    # Small delay to prevent rate limiting
                    await asyncio.sleep(1)
        
        # Analyze experiment results
        analysis = await self._analyze_experiment_results(experiment_results, experiment_name)
        
        # Save comprehensive results
        experiment_data = {
            "experiment_id": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "interrogator_types": interrogator_types,
                "temperature_range": temperature_range,
                "tests_per_temperature": tests_per_temperature,
                "include_human_proxy": include_human_proxy,
                "total_tests": total_tests
            },
            "individual_results": experiment_results,
            "analysis": analysis
        }
        
        await self._save_experiment_results(experiment_data)
        
        logger.info(f"Completed temperature matrix experiment: {experiment_name}")
        logger.info(f"Overall accuracy: {analysis['overall_accuracy']:.1%}")
        
        return experiment_data
    
    def _create_interrogator(self, interrogator_type: str) -> InterrogatorAgent:
        """Create an interrogator agent of the specified type."""
        
        from autogen_ext.agents.turing_test._interrogator_agent import create_interrogator
        
        return create_interrogator(
            interrogator_type=interrogator_type,
            model_client=self.model_client,
            max_rounds=self.max_rounds
        )
    
    def _create_subject(
        self, 
        subject_type: str, 
        temperature: float, 
        personality: Optional[str]
    ) -> SubjectAgent:
        """Create a subject agent of the specified type."""
        
        from autogen_ext.agents.turing_test._subject_agent import create_ai_subject, create_human_proxy_subject
        
        if subject_type == "ai":
            return create_ai_subject(
                model_client=self.model_client,
                temperature=temperature
            )
        else:
            personality_type = personality if personality != "random" else None
            return create_human_proxy_subject(
                model_client=self.model_client,
                personality_type=personality_type
            )
    
    async def _run_conversation(
        self, 
        interrogator: InterrogatorAgent, 
        subject: SubjectAgent,
        test_id: str
    ) -> Dict[str, Any]:
        """Run the actual conversation between interrogator and subject."""
        
        # Create team for conversation
        team = RoundRobinGroupChat([interrogator, subject])
        
        # Initial message to start conversation
        initial_task = "Begin the Turing Test conversation. Interrogator, please start with your first question."
        
        try:
            # Run conversation with message limit
            result = await team.run(
                task=initial_task,
                termination_condition=MaxMessageCondition(max_messages=self.max_rounds * 2)
            )
            
            return {
                "status": "completed",
                "messages": [msg.__dict__ for msg in result.messages] if hasattr(result, 'messages') else [],
                "task_result": str(result) if result else "No result"
            }
            
        except Exception as e:
            logger.error(f"Error during conversation {test_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "messages": []
            }
    
    async def _analyze_experiment_results(
        self, 
        results: List[Dict[str, Any]], 
        experiment_name: str
    ) -> Dict[str, Any]:
        """Analyze results from a complete experiment."""
        
        if not results:
            return {"error": "No results to analyze"}
        
        # Basic statistics
        total_tests = len(results)
        correct_identifications = sum(1 for r in results if r["evaluation"]["correct_identification"])
        overall_accuracy = correct_identifications / total_tests
        
        # Performance by temperature
        temp_performance = {}
        for result in results:
            temp = result["configuration"]["subject_temperature"]
            if temp not in temp_performance:
                temp_performance[temp] = {"total": 0, "correct": 0, "ai_tests": 0, "human_tests": 0}
            
            temp_performance[temp]["total"] += 1
            if result["evaluation"]["correct_identification"]:
                temp_performance[temp]["correct"] += 1
            
            if result["configuration"]["subject_type"] == "ai":
                temp_performance[temp]["ai_tests"] += 1
            else:
                temp_performance[temp]["human_tests"] += 1
        
        # Calculate accuracy by temperature
        for temp in temp_performance:
            data = temp_performance[temp]
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
        
        # Performance by interrogator type
        interrogator_performance = {}
        for result in results:
            interr_type = result["configuration"]["interrogator_type"]
            if interr_type not in interrogator_performance:
                interrogator_performance[interr_type] = {"total": 0, "correct": 0}
            
            interrogator_performance[interr_type]["total"] += 1
            if result["evaluation"]["correct_identification"]:
                interrogator_performance[interr_type]["correct"] += 1
        
        for interr_type in interrogator_performance:
            data = interrogator_performance[interr_type]
            data["accuracy"] = data["correct"] / data["total"] if data["total"] > 0 else 0
        
        # Average scores
        avg_confidence = sum(r["evaluation"]["confidence_score"] for r in results) / total_tests
        avg_authenticity = sum(r["evaluation"]["authenticity_score"] for r in results) / total_tests
        avg_quality = sum(r["evaluation"]["conversation_quality"] for r in results) / total_tests
        
        return {
            "experiment_name": experiment_name,
            "total_tests": total_tests,
            "correct_identifications": correct_identifications,
            "overall_accuracy": overall_accuracy,
            "performance_by_temperature": temp_performance,
            "performance_by_interrogator": interrogator_performance,
            "average_scores": {
                "confidence": avg_confidence,
                "authenticity": avg_authenticity,
                "quality": avg_quality
            },
            "best_temperature": max(temp_performance.keys(), key=lambda t: temp_performance[t]["accuracy"]),
            "best_interrogator": max(interrogator_performance.keys(), key=lambda i: interrogator_performance[i]["accuracy"])
        }
    
    async def _save_test_result(self, result: Dict[str, Any]):
        """Save individual test result to file."""
        
        filename = f"{result['test_id']}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    async def _save_experiment_results(self, experiment_data: Dict[str, Any]):
        """Save complete experiment results."""
        
        experiment_id = experiment_data["experiment_id"]
        
        # Save JSON results
        json_filename = f"{experiment_id}_results.json"
        json_filepath = self.results_dir / json_filename
        
        with open(json_filepath, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        # Save CSV summary
        csv_filename = f"{experiment_id}_summary.csv"
        csv_filepath = self.results_dir / csv_filename
        
        with open(csv_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "test_id", "interrogator_type", "subject_type", "temperature",
                "correct_identification", "confidence_score", "authenticity_score",
                "conversation_quality", "total_rounds", "duration_seconds"
            ])
            
            for result in experiment_data["individual_results"]:
                writer.writerow([
                    result["test_id"],
                    result["configuration"]["interrogator_type"],
                    result["configuration"]["subject_type"],
                    result["configuration"]["subject_temperature"],
                    result["evaluation"]["correct_identification"],
                    result["evaluation"]["confidence_score"],
                    result["evaluation"]["authenticity_score"],
                    result["evaluation"]["conversation_quality"],
                    result["evaluation"]["total_rounds"],
                    result["evaluation"]["conversation_duration"]
                ])
        
        logger.info(f"Experiment results saved: {json_filepath} and {csv_filepath}")

# Convenience function for quick experiments
async def run_quick_turing_test(
    model_client: ChatCompletionClient,
    interrogator_type: str = "investigative",
    subject_type: str = "ai",
    temperature: float = 0.7,
    max_rounds: int = 5
) -> Dict[str, Any]:
    """
    Run a quick single Turing Test for testing purposes.
    
    Args:
        model_client: Chat completion client
        interrogator_type: Type of interrogator
        subject_type: "ai" or "human_proxy"
        temperature: Subject temperature
        max_rounds: Maximum conversation rounds
        
    Returns:
        Test results
    """
    
    team = TuringTestTeam(
        model_client=model_client,
        max_rounds=max_rounds,
        results_dir="quick_test_results"
    )
    
    return await team.run_single_test(
        interrogator_type=interrogator_type,
        subject_type=subject_type,
        subject_temperature=temperature
    )