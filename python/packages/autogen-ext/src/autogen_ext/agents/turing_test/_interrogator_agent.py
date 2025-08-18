"""
Interrogator Agent for AutoGen Turing Test Extension

The InterrogatorAgent conducts Turing Test conversations by asking questions
designed to distinguish between human and AI participants.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import asyncio
import logging
from datetime import datetime
import json

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_core.models import ChatCompletionClient

logger = logging.getLogger(__name__)

class InterrogatorAgent(AssistantAgent):
    """
    An agent that conducts Turing Test conversations by asking probing questions
    to determine if participants are human or AI.
    """
    
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        system_message: Optional[str] = None,
        max_rounds: int = 10,
        conversation_style: str = "investigative",
        question_categories: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the InterrogatorAgent.
        
        Args:
            name: Agent name
            model_client: Chat completion client (Azure OpenAI, etc.)
            system_message: Custom system prompt for the interrogator
            max_rounds: Maximum number of question rounds
            conversation_style: Style of questioning ("investigative", "casual", "philosophical")
            question_categories: Categories of questions to focus on
            **kwargs: Additional arguments passed to AssistantAgent
        """
        
        if system_message is None:
            system_message = self._create_default_system_message(conversation_style, question_categories)
        
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
            **kwargs
        )
        
        self.max_rounds = max_rounds
        self.conversation_style = conversation_style
        self.question_categories = question_categories or self._default_question_categories()
        self.current_round = 0
        self.conversation_log = []
        self.start_time = None
        
    def _create_default_system_message(self, style: str, categories: Optional[List[str]]) -> str:
        """Create a default system message for the interrogator."""
        
        base_prompt = """You are an experienced interrogator conducting a Turing Test. Your goal is to determine whether the participant you're conversing with is human or AI through strategic questioning.

CORE OBJECTIVES:
1. Ask thoughtful, probing questions that might reveal human vs AI characteristics
2. Follow up on responses to dig deeper into reasoning patterns
3. Look for signs of genuine human experience, emotion, and creativity
4. Detect potential AI patterns like overly structured responses or lack of personal experience
5. Maintain a natural conversation flow while gathering evidence

QUESTIONING STRATEGY:"""
        
        if style == "investigative":
            strategy = """
- Ask about personal experiences and memories
- Probe emotional responses and subjective feelings  
- Test creative thinking and abstract reasoning
- Explore cultural knowledge and lived experiences
- Challenge with unexpected or ambiguous scenarios"""
        
        elif style == "casual":
            strategy = """
- Engage in natural, friendly conversation
- Ask about daily life and personal preferences
- Explore hobbies, relationships, and experiences
- Use humor and informal language
- Look for authentic personal details and reactions"""
        
        elif style == "philosophical":
            strategy = """
- Explore abstract concepts and ethical reasoning
- Discuss consciousness, identity, and subjective experience
- Ask about beliefs, values, and meaning-making
- Test nuanced understanding of complex topics
- Probe intuitive vs analytical thinking patterns"""
        
        else:
            strategy = """
- Adapt questioning style based on participant responses
- Use a mix of personal, creative, and analytical questions
- Follow conversational threads naturally
- Look for authenticity markers in responses"""
        
        categories_text = ""
        if categories:
            categories_text = f"\nFOCUS AREAS: {', '.join(categories)}"
        
        closing = """
CONVERSATION GUIDELINES:
- Ask one clear question per turn
- Keep questions engaging and natural
- Avoid being overly aggressive or robotic
- Build on previous responses
- Stay within the conversation context
- Make your final determination based on accumulated evidence

Begin with a natural greeting and your first question."""
        
        return base_prompt + strategy + categories_text + closing
    
    def _default_question_categories(self) -> List[str]:
        """Default categories for Turing Test questions."""
        return [
            "personal_experience",
            "emotional_intelligence", 
            "creativity",
            "cultural_knowledge",
            "abstract_reasoning",
            "sensory_experience",
            "social_dynamics",
            "unexpected_scenarios"
        ]
    
    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: Any) -> Response:
        """
        Process incoming messages and generate interrogator response.
        """
        if self.start_time is None:
            self.start_time = datetime.now()
        
        # Log the conversation
        latest_message = messages[-1] if messages else None
        if latest_message:
            self.conversation_log.append({
                "timestamp": datetime.now().isoformat(),
                "round": self.current_round,
                "participant_message": latest_message.content,
                "message_type": type(latest_message).__name__
            })
        
        # Check if we've reached max rounds
        if self.current_round >= self.max_rounds:
            return self._generate_final_assessment()
        
        # Generate response using parent class
        response = await super().on_messages(messages, cancellation_token)
        
        # Log our response
        if isinstance(response, Response):
            self.conversation_log.append({
                "timestamp": datetime.now().isoformat(),
                "round": self.current_round,
                "interrogator_question": response.chat_message.content,
                "response_type": type(response).__name__
            })
        
        self.current_round += 1
        return response
    
    def _generate_final_assessment(self) -> Response:
        """Generate final assessment of whether participant is human or AI."""
        
        assessment_prompt = """Based on the conversation, make your final determination:

ASSESSMENT CRITERIA:
- Authenticity of personal experiences and emotions
- Naturalness of language and response patterns  
- Creativity and original thinking
- Cultural knowledge and lived experience
- Response time and conversation flow
- Consistency and depth of responses

Provide your verdict as either "HUMAN" or "AI" followed by your reasoning."""
        
        assessment_message = TextMessage(
            content=assessment_prompt,
            source=self.name
        )
        
        # Create summary of conversation for context
        conversation_summary = self._create_conversation_summary()
        
        final_message = TextMessage(
            content=f"FINAL ASSESSMENT:\n\n{conversation_summary}\n\n{assessment_prompt}",
            source=self.name
        )
        
        return Response(chat_message=final_message)
    
    def _create_conversation_summary(self) -> str:
        """Create a summary of the conversation for final assessment."""
        
        if not self.conversation_log:
            return "No conversation recorded."
        
        duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        summary = f"""CONVERSATION SUMMARY:
Duration: {duration:.1f} seconds
Total Rounds: {self.current_round}
Style: {self.conversation_style}

KEY EXCHANGES:"""
        
        for i, entry in enumerate(self.conversation_log):
            if "participant_message" in entry:
                summary += f"\nRound {entry['round']}: {entry['participant_message'][:100]}..."
            if "interrogator_question" in entry:
                summary += f"\nQ{entry['round']}: {entry['interrogator_question'][:100]}..."
        
        return summary
    
    def get_conversation_data(self) -> Dict[str, Any]:
        """Get structured conversation data for analysis."""
        
        return {
            "interrogator_name": self.name,
            "conversation_style": self.conversation_style,
            "question_categories": self.question_categories,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "total_rounds": self.current_round,
            "max_rounds": self.max_rounds,
            "conversation_log": self.conversation_log,
            "status": "completed" if self.current_round >= self.max_rounds else "in_progress"
        }
    
    def reset_conversation(self):
        """Reset the interrogator for a new conversation."""
        self.current_round = 0
        self.conversation_log = []
        self.start_time = None
        logger.info(f"Interrogator {self.name} reset for new conversation")

# Convenience function for creating common interrogator types
def create_interrogator(
    interrogator_type: str,
    model_client: ChatCompletionClient,
    name: Optional[str] = None,
    **kwargs
) -> InterrogatorAgent:
    """
    Create a pre-configured interrogator agent.
    
    Args:
        interrogator_type: Type of interrogator ("casual", "investigative", "philosophical", "creative")
        model_client: Chat completion client
        name: Agent name (defaults to type-based name)
        **kwargs: Additional arguments
    
    Returns:
        Configured InterrogatorAgent
    """
    
    if name is None:
        name = f"{interrogator_type.title()}Interrogator"
    
    type_configs = {
        "casual": {
            "conversation_style": "casual",
            "question_categories": ["personal_experience", "daily_life", "preferences", "social_dynamics"]
        },
        "investigative": {
            "conversation_style": "investigative", 
            "question_categories": ["personal_experience", "emotional_intelligence", "cultural_knowledge"]
        },
        "philosophical": {
            "conversation_style": "philosophical",
            "question_categories": ["abstract_reasoning", "consciousness", "ethics", "meaning_making"]
        },
        "creative": {
            "conversation_style": "investigative",
            "question_categories": ["creativity", "imagination", "artistic_expression", "unexpected_scenarios"]
        }
    }
    
    config = type_configs.get(interrogator_type, type_configs["investigative"])
    config.update(kwargs)
    
    return InterrogatorAgent(
        name=name,
        model_client=model_client,
        **config
    )