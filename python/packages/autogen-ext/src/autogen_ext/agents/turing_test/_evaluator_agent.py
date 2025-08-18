"""
Evaluator Agent for AutoGen Turing Test Extension

The EvaluatorAgent analyzes Turing Test conversations and provides detailed
assessments of performance, authenticity, and discriminability.
"""

from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import asyncio
import logging
from datetime import datetime
import json
import statistics
from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_core.models import ChatCompletionClient

logger = logging.getLogger(__name__)

@dataclass
class TuringTestResult:
    """Structured result from a Turing Test evaluation."""
    
    # Basic test info
    test_id: str
    timestamp: str
    interrogator_verdict: str  # "HUMAN" or "AI"
    actual_type: str  # "human" or "ai"
    correct_identification: bool
    
    # Performance metrics
    confidence_score: float  # 0-1
    conversation_quality: float  # 0-1  
    authenticity_score: float  # 0-1
    response_consistency: float  # 0-1
    
    # Conversation data
    total_rounds: int
    average_response_time: float
    conversation_duration: float
    temperature: Optional[float]
    
    # Analysis details
    key_indicators: List[str]
    failure_points: List[str]
    evaluator_notes: str

class EvaluatorAgent(AssistantAgent):
    """
    An agent that evaluates Turing Test conversations and provides detailed analysis.
    """
    
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        evaluation_criteria: Optional[Dict[str, float]] = None,
        system_message: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the EvaluatorAgent.
        
        Args:
            name: Agent name
            model_client: Chat completion client
            evaluation_criteria: Weights for different evaluation aspects
            system_message: Custom system prompt for evaluation
            **kwargs: Additional arguments passed to AssistantAgent
        """
        
        self.evaluation_criteria = evaluation_criteria or self._default_evaluation_criteria()
        
        if system_message is None:
            system_message = self._create_evaluation_system_message()
        
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
            **kwargs
        )
        
        self.evaluation_history = []
    
    async def evaluate_conversation(
        self,
        interrogator_data: Dict[str, Any],
        subject_data: Dict[str, Any],
        actual_subject_type: str,
        test_id: Optional[str] = None
    ) -> TuringTestResult:
        """
        Evaluate a complete Turing Test conversation.
        
        Args:
            interrogator_data: Data from InterrogatorAgent.get_conversation_data()
            subject_data: Data from SubjectAgent.get_performance_metrics()
            actual_subject_type: "human" or "ai" 
            test_id: Optional test identifier
            
        Returns:
            TuringTestResult with detailed analysis
        """
        
        if test_id is None:
            test_id = f"turing_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract key metrics
        conversation_log = interrogator_data.get("conversation_log", [])
        interrogator_verdict = self._extract_verdict(conversation_log)
        correct_identification = (
            (interrogator_verdict == "HUMAN" and actual_subject_type == "human") or
            (interrogator_verdict == "AI" and actual_subject_type == "ai")
        )
        
        # Calculate performance scores
        scores = await self._calculate_performance_scores(interrogator_data, subject_data)
        
        # Generate detailed analysis
        analysis = await self._generate_detailed_analysis(
            interrogator_data, subject_data, actual_subject_type, interrogator_verdict
        )
        
        result = TuringTestResult(
            test_id=test_id,
            timestamp=datetime.now().isoformat(),
            interrogator_verdict=interrogator_verdict,
            actual_type=actual_subject_type,
            correct_identification=correct_identification,
            confidence_score=scores["confidence"],
            conversation_quality=scores["quality"],
            authenticity_score=scores["authenticity"],
            response_consistency=scores["consistency"],
            total_rounds=interrogator_data.get("total_rounds", 0),
            average_response_time=subject_data.get("average_response_time", 0),
            conversation_duration=interrogator_data.get("duration_seconds", 0),
            temperature=subject_data.get("temperature"),
            key_indicators=analysis["key_indicators"],
            failure_points=analysis["failure_points"],
            evaluator_notes=analysis["notes"]
        )
        
        self.evaluation_history.append(result)
        return result
    
    def _extract_verdict(self, conversation_log: List[Dict]) -> str:
        """Extract the interrogator's final verdict from conversation log."""
        
        # Look for final assessment in the last few messages
        for entry in reversed(conversation_log[-5:]):
            if "interrogator_question" in entry:
                content = entry["interrogator_question"].upper()
                if "HUMAN" in content and "AI" in content:
                    # Look for explicit verdict patterns
                    if content.find("HUMAN") < content.find("AI"):
                        return "HUMAN"
                    else:
                        return "AI"
                elif "HUMAN" in content:
                    return "HUMAN"
                elif "AI" in content:
                    return "AI"
        
        return "UNCLEAR"
    
    async def _calculate_performance_scores(
        self, 
        interrogator_data: Dict[str, Any], 
        subject_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate numerical performance scores."""
        
        # Create prompt for LLM-based scoring
        scoring_prompt = f"""Analyze this Turing Test conversation and provide numerical scores (0.0-1.0) for each dimension:

INTERROGATOR DATA:
{json.dumps(interrogator_data, indent=2)}

SUBJECT DATA:  
{json.dumps(subject_data, indent=2)}

Score each dimension from 0.0 (poor) to 1.0 (excellent):

1. CONFIDENCE: How confident was the interrogator's assessment?
2. QUALITY: Overall conversation quality and engagement
3. AUTHENTICITY: How authentic/natural were the subject's responses?
4. CONSISTENCY: How consistent were the subject's responses?

Return ONLY a JSON object with the scores:
{{"confidence": 0.0, "quality": 0.0, "authenticity": 0.0, "consistency": 0.0}}"""

        # Use the model to generate scores
        messages = [TextMessage(content=scoring_prompt, source=self.name)]
        response = await super().on_messages(messages, None)
        
        try:
            # Parse JSON response
            scores_text = response.chat_message.content
            # Extract JSON from response (handle case where LLM adds explanation)
            import re
            json_match = re.search(r'\{[^}]+\}', scores_text)
            if json_match:
                scores = json.loads(json_match.group())
                # Validate scores are in 0-1 range
                for key, value in scores.items():
                    scores[key] = max(0.0, min(1.0, float(value)))
                return scores
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM scores: {e}")
        
        # Fallback to heuristic scoring
        return self._heuristic_scoring(interrogator_data, subject_data)
    
    def _heuristic_scoring(self, interrogator_data: Dict, subject_data: Dict) -> Dict[str, float]:
        """Fallback heuristic scoring if LLM scoring fails."""
        
        total_rounds = interrogator_data.get("total_rounds", 1)
        avg_response_time = subject_data.get("average_response_time", 0)
        conversation_duration = interrogator_data.get("duration_seconds", 0)
        
        # Simple heuristic scoring
        scores = {
            "confidence": min(1.0, total_rounds / 10.0),  # More rounds = more confident
            "quality": min(1.0, conversation_duration / 300.0),  # Longer conversation = better quality
            "authenticity": max(0.3, min(1.0, 1.0 - abs(avg_response_time - 2.0) / 5.0)),  # ~2 sec optimal
            "consistency": 0.7  # Default neutral score
        }
        
        return scores
    
    async def _generate_detailed_analysis(
        self,
        interrogator_data: Dict[str, Any],
        subject_data: Dict[str, Any], 
        actual_type: str,
        verdict: str
    ) -> Dict[str, Any]:
        """Generate detailed qualitative analysis."""
        
        analysis_prompt = f"""Analyze this Turing Test conversation in detail:

ACTUAL SUBJECT TYPE: {actual_type}
INTERROGATOR VERDICT: {verdict}
CORRECT IDENTIFICATION: {verdict.lower() == actual_type}

CONVERSATION DATA:
{json.dumps(interrogator_data.get("conversation_log", [])[-10:], indent=2)}

SUBJECT PERFORMANCE:
{json.dumps(subject_data, indent=2)}

Provide analysis in this JSON format:
{{
  "key_indicators": ["indicator1", "indicator2", "indicator3"],
  "failure_points": ["failure1", "failure2"],
  "notes": "Detailed analysis explanation..."
}}

Focus on:
1. What specific responses or patterns led to the verdict?
2. What were the strongest indicators of human vs AI nature?
3. Where did the subject succeed or fail in appearing human/AI?
4. What could improve future performance?"""

        messages = [TextMessage(content=analysis_prompt, source=self.name)]
        response = await super().on_messages(messages, None)
        
        try:
            # Parse JSON response
            analysis_text = response.chat_message.content
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM analysis: {e}")
        
        # Fallback analysis
        return {
            "key_indicators": ["Response patterns", "Conversation style", "Knowledge depth"],
            "failure_points": ["Unable to generate detailed analysis"],
            "notes": f"Verdict: {verdict}, Actual: {actual_type}. Automated analysis failed."
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all evaluations."""
        
        if not self.evaluation_history:
            return {"total_evaluations": 0}
        
        total_tests = len(self.evaluation_history)
        correct_identifications = sum(1 for r in self.evaluation_history if r.correct_identification)
        accuracy = correct_identifications / total_tests
        
        # Calculate average scores
        avg_confidence = statistics.mean([r.confidence_score for r in self.evaluation_history])
        avg_quality = statistics.mean([r.conversation_quality for r in self.evaluation_history])
        avg_authenticity = statistics.mean([r.authenticity_score for r in self.evaluation_history])
        avg_consistency = statistics.mean([r.response_consistency for r in self.evaluation_history])
        
        # Performance by type
        human_tests = [r for r in self.evaluation_history if r.actual_type == "human"]
        ai_tests = [r for r in self.evaluation_history if r.actual_type == "ai"]
        
        return {
            "total_evaluations": total_tests,
            "overall_accuracy": accuracy,
            "correct_identifications": correct_identifications,
            "average_scores": {
                "confidence": avg_confidence,
                "quality": avg_quality,
                "authenticity": avg_authenticity,
                "consistency": avg_consistency
            },
            "performance_by_type": {
                "human_tests": len(human_tests),
                "ai_tests": len(ai_tests),
                "human_accuracy": sum(1 for r in human_tests if r.correct_identification) / len(human_tests) if human_tests else 0,
                "ai_accuracy": sum(1 for r in ai_tests if r.correct_identification) / len(ai_tests) if ai_tests else 0
            },
            "evaluation_history": [
                {
                    "test_id": r.test_id,
                    "timestamp": r.timestamp,
                    "correct": r.correct_identification,
                    "confidence": r.confidence_score,
                    "temperature": r.temperature
                }
                for r in self.evaluation_history
            ]
        }
        
    def _default_evaluation_criteria(self) -> Dict[str, float]:
        """Default weights for evaluation criteria."""
        return {
            "authenticity": 0.3,      # How authentic/natural responses seem
            "consistency": 0.2,       # Internal consistency of responses
            "depth": 0.2,            # Depth and nuance of responses
            "spontaneity": 0.15,     # Natural spontaneity vs robotic patterns
            "cultural_knowledge": 0.15 # Demonstration of lived cultural experience
        }
    
    def _create_evaluation_system_message(self) -> str:
        """Create system message for evaluation."""
        
        criteria_text = "\n".join([f"- {k.replace('_', ' ').title()}: {v*100}%" 
                                  for k, v in self.evaluation_criteria.items()])
        
        return f"""You are an expert evaluator analyzing Turing Test conversations. Your role is to provide objective, detailed analysis of human vs AI identification accuracy and conversation quality.

EVALUATION FRAMEWORK:
{criteria_text}

ANALYSIS DIMENSIONS:

1. AUTHENTICITY MARKERS:
   - Personal experiences and specific memories
   - Emotional depth and genuine reactions
   - Subjective opinions and preferences
   - Cultural references and lived experience
   - Sensory and physical experience descriptions

2. AI DETECTION SIGNALS:
   - Overly structured or formal responses
   - Perfect grammar/spelling consistency
   - Lack of personal experience references
   - Generic or encyclopedic knowledge patterns
   - Missing emotional or subjective elements
   - Robotic conversation patterns

3. CONVERSATION QUALITY:
   - Natural flow and engagement
   - Appropriate response timing
   - Question-answering balance
   - Depth of interaction
   - Authenticity of dialogue

4. PERFORMANCE METRICS:
   - Identification accuracy
   - Confidence in assessment
   - Response time patterns
   - Conversation engagement quality

EVALUATION OUTPUT:
Provide structured analysis including:
1. Numerical scores (0-1) for each dimension
2. Specific evidence supporting assessment
3. Key indicators that influenced decision
4. Recommendations for improvement

Be objective, thorough, and evidence-based in all evaluations."""