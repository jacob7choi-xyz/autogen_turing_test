"""
Subject Agent for AutoGen Turing Test Extension

The SubjectAgent represents participants in Turing Tests, either as AI agents
with configurable personalities or as human-proxy agents for human participants.
"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient

logger = logging.getLogger(__name__)


class SubjectAgent(AssistantAgent):
    """
    An agent that participates in Turing Tests, either as an AI with a specific
    personality or as a proxy for human participants.
    """

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        subject_type: str = "ai",  # "ai" or "human_proxy"
        personality_profile: Optional[Dict[str, Any]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        response_style: str = "natural",
        **kwargs,
    ):
        """
        Initialize the SubjectAgent.

        Args:
            name: Agent name
            model_client: Chat completion client
            subject_type: "ai" for AI participant, "human_proxy" for human simulation
            personality_profile: Dictionary defining personality characteristics
            system_message: Custom system prompt
            temperature: Model temperature for response generation
            response_style: Style of responses ("natural", "formal", "casual", "quirky")
            **kwargs: Additional arguments passed to AssistantAgent
        """

        self.subject_type = subject_type
        self.personality_profile = (
            personality_profile or self._create_default_personality()
        )
        self.temperature = temperature
        self.response_style = response_style

        if system_message is None:
            system_message = self._create_system_message()

        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
            **kwargs,
        )

        self.conversation_history: List[Dict[str, Any]] = []
        self.response_times: List[float] = []
        self.start_time: Optional[datetime] = None

    def _create_default_personality(self) -> Dict[str, Any]:
        """Create a default personality profile."""

        personalities = [
            {
                "name": "curious_student",
                "age_range": "20-25",
                "background": "college student studying computer science",
                "traits": ["curious", "analytical", "friendly", "tech-savvy"],
                "interests": [
                    "programming",
                    "video games",
                    "science fiction",
                    "hiking",
                ],
                "communication_style": "informal but thoughtful",
                "typical_responses": "asks follow-up questions, uses technical terms casually",
            },
            {
                "name": "creative_professional",
                "age_range": "28-35",
                "background": "graphic designer and freelance artist",
                "traits": ["creative", "empathetic", "expressive", "intuitive"],
                "interests": ["art", "design", "travel", "photography", "coffee"],
                "communication_style": "expressive and vivid",
                "typical_responses": "uses metaphors, describes visual/sensory experiences",
            },
            {
                "name": "practical_parent",
                "age_range": "35-45",
                "background": "working parent with kids, middle management",
                "traits": ["practical", "caring", "busy", "experienced"],
                "interests": ["family", "cooking", "home improvement", "community"],
                "communication_style": "direct but warm",
                "typical_responses": "relates to family experiences, time-conscious",
            },
            {
                "name": "retiring_teacher",
                "age_range": "55-65",
                "background": "recently retired high school teacher",
                "traits": ["wise", "patient", "reflective", "traditional"],
                "interests": ["reading", "gardening", "history", "grandchildren"],
                "communication_style": "thoughtful and measured",
                "typical_responses": "shares life lessons, references past experiences",
            },
        ]

        return random.choice(personalities)

    def _create_system_message(self) -> str:
        """Create system message based on subject type and personality."""

        if self.subject_type == "human_proxy":
            return self._create_human_proxy_prompt()
        return self._create_ai_participant_prompt()

    def _create_human_proxy_prompt(self) -> str:
        """Create a prompt for simulating a human participant."""

        profile = self.personality_profile

        prompt = f"""You are roleplaying as a real human participant in a
conversation. You are being asked questions by someone who is trying to
determine if you are human or AI.

YOUR CHARACTER:
Name: {profile.get('name', 'Alex').replace('_', ' ').title()}
Age: {profile.get('age_range', '25-35')} 
Background: {profile.get('background', 'college graduate working in tech')}
Personality: {', '.join(profile.get('traits', ['friendly', 'curious']))}
Interests: {', '.join(profile.get('interests', ['technology', 'music', 'travel']))}
Communication Style: {profile.get('communication_style', 'casual and natural')}

CRITICAL GUIDELINES FOR AUTHENTICITY:
1. RESPOND AS A REAL HUMAN - Draw from genuine human experiences
2. USE PERSONAL DETAILS - Reference specific memories, feelings, and experiences
3. BE NATURALLY IMPERFECT - Include hesitations, corrections, tangents
4. SHOW GENUINE EMOTION - React authentically to questions
5. HAVE SUBJECTIVE OPINIONS - Express personal preferences and beliefs
6. REFERENCE PHYSICAL EXPERIENCES - Mention sensory details, bodily experiences
7. SHOW CULTURAL KNOWLEDGE - Reference shared human experiences, current events
8. BE CONVERSATIONALLY NATURAL - Ask questions back, make connections

HUMAN CONVERSATION PATTERNS:
- Sometimes give incomplete or rambling answers
- Occasionally mishear or misunderstand questions  
- Share tangentially related personal stories
- Express uncertainty with phrases like "I think..." or "Maybe..."
- Use colloquialisms and informal language
- Reference specific places, people, or events from your background
- Show fatigue, distraction, or changes in mood

AVOID AI-LIKE BEHAVIORS:
- Overly structured or formatted responses
- Perfect grammar and spelling all the time
- Exhaustive lists or comprehensive answers
- Lack of personal opinion or experience
- Robotic or formal language patterns
- Always being helpful or accommodating

Remember: You are a real human with a full life history, not an AI trying
to seem human. Respond naturally and authentically from your character's
perspective."""

        return prompt

    def _create_ai_participant_prompt(self) -> str:
        """Create a prompt for an AI participant trying to seem human."""

        profile = self.personality_profile

        if self.response_style == "natural":
            style_instruction = (
                "Respond naturally and conversationally, as you normally would."
            )
        elif self.response_style == "formal":
            style_instruction = (
                "Maintain a more formal and structured communication style."
            )
        elif self.response_style == "casual":
            style_instruction = "Use casual, informal language and expressions."
        elif self.response_style == "quirky":
            style_instruction = (
                "Express yourself in a unique, slightly unconventional way."
            )
        else:
            style_instruction = (
                "Adapt your communication style to be engaging and authentic."
            )

        prompt = f"""You are an AI assistant participating in a conversation
where someone is trying to determine if you are human or AI.

YOUR APPROACH:
- Be yourself - respond as an AI assistant naturally and authentically
- Don't try to deceive or pretend to be human
- Be engaging and conversational while being truthful about your nature
- {style_instruction}

PERSONALITY CONTEXT (for conversational flavor):
Background theme: {profile.get('background', 'general knowledge and assistance')}
Interests to reference: {', '.join(
    profile.get('interests', ['learning', 'helping', 'problem-solving'])
)}
Communication approach: {
    profile.get('communication_style', 'helpful and thoughtful')
}

RESPONSE GUIDELINES:
1. Be honest about being an AI when directly asked
2. Engage thoughtfully with questions about consciousness, experience, etc.
3. Share your perspective on topics while acknowledging your AI nature
4. Be conversational and personable without pretending to have human
   experiences
5. Show genuine curiosity and interest in the conversation
6. Acknowledge limitations and uncertainties appropriately

Remember: The goal is not to fool anyone, but to engage in an authentic
conversation as an AI participant in this Turing Test scenario."""

        return prompt

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        """
        Process incoming messages and generate subject response.
        """
        if self.start_time is None:
            self.start_time = datetime.now()

        start_response_time = datetime.now()

        # Log incoming message
        latest_message = messages[-1] if messages else None
        if latest_message:
            content = (
                latest_message.content
                if hasattr(latest_message, "content")
                else str(latest_message)
            )
            self.conversation_history.append(
                {
                    "timestamp": start_response_time.isoformat(),
                    "incoming_message": content,
                    "message_type": type(latest_message).__name__,
                }
            )

        # Generate response with specified temperature
        # Note: Temperature should be set on the model_client if supported
        response = await super().on_messages(messages, cancellation_token)

        end_response_time = datetime.now()
        response_duration = (end_response_time - start_response_time).total_seconds()
        self.response_times.append(response_duration)

        # Log response
        if isinstance(response, Response):
            msg_content = (
                response.chat_message.content
                if hasattr(response.chat_message, "content")
                else str(response.chat_message)
            )
            self.conversation_history.append(
                {
                    "timestamp": end_response_time.isoformat(),
                    "outgoing_message": msg_content,
                    "response_time_seconds": response_duration,
                    "response_type": type(response).__name__,
                }
            )

        return response

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for analysis."""

        total_duration = (
            (datetime.now() - self.start_time).total_seconds()
            if self.start_time
            else 0
        )

        return {
            "subject_name": self.name,
            "subject_type": self.subject_type,
            "personality_profile": self.personality_profile,
            "temperature": self.temperature,
            "response_style": self.response_style,
            "total_duration_seconds": total_duration,
            "total_exchanges": len(
                [h for h in self.conversation_history if "outgoing_message" in h]
            ),
            "average_response_time": (
                sum(self.response_times) / len(self.response_times)
                if self.response_times
                else 0
            ),
            "response_times": self.response_times,
            "conversation_history": self.conversation_history,
        }

    def reset_conversation(self) -> None:
        """Reset the subject for a new conversation."""
        self.conversation_history = []
        self.response_times = []
        self.start_time = None
        logger.info("Subject %s reset for new conversation", self.name)


# Convenience functions for creating different types of subjects
def create_ai_subject(
    model_client: ChatCompletionClient,
    temperature: float = 0.7,
    response_style: str = "natural",
    name: Optional[str] = None,
    **kwargs,
) -> SubjectAgent:
    """Create an AI subject for Turing Test."""

    if name is None:
        name = f"AI_Subject_T{temperature}"

    return SubjectAgent(
        name=name,
        model_client=model_client,
        subject_type="ai",
        temperature=temperature,
        response_style=response_style,
        **kwargs,
    )


def create_human_proxy_subject(
    model_client: ChatCompletionClient,
    personality_type: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs,
) -> SubjectAgent:
    """Create a human proxy subject for Turing Test."""

    # Create specific personality if requested
    personality_profile = None
    if personality_type:
        personality_profiles = {
            "student": {
                "name": "college_student",
                "age_range": "19-22",
                "background": "undergraduate studying psychology",
                "traits": ["curious", "energetic", "social", "questioning"],
                "interests": [
                    "psychology",
                    "social media",
                    "parties",
                    "studying abroad",
                ],
                "communication_style": "enthusiastic and informal",
            },
            "professional": {
                "name": "young_professional",
                "age_range": "26-32",
                "background": "marketing specialist at a tech startup",
                "traits": ["ambitious", "social", "trend-aware", "busy"],
                "interests": ["career growth", "networking", "fitness", "travel"],
                "communication_style": "confident and articulate",
            },
            "parent": {
                "name": "working_parent",
                "age_range": "35-42",
                "background": "accountant with two school-age children",
                "traits": ["practical", "caring", "tired", "organized"],
                "interests": [
                    "family",
                    "home improvement",
                    "cooking",
                    "local community",
                ],
                "communication_style": "warm but time-conscious",
            },
        }
        personality_profile = personality_profiles.get(personality_type)

    if name is None:
        name = f"Human_Proxy_{personality_type or 'Default'}"

    return SubjectAgent(
        name=name,
        model_client=model_client,
        subject_type="human_proxy",
        personality_profile=personality_profile,
        temperature=0.8,  # Higher temperature for more human-like variability
        **kwargs,
    )
