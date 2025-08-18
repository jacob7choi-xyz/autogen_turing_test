#!/usr/bin/env python3
"""
Faithful Recreation of Original Turing Test Research
==================================================

This script recreates the exact methodology from Jacob Choi's original research
with Mark Encarnación, but using proper AutoGen v0.4 architecture and
software engineering practices.

Author: Jacob J. Choi (Colby College, CS: AI)
Supervisor: Mark Encarnación (Microsoft Research)
Rebuilt: AutoGen v0.4 with proper engineering practices

Key Features Preserved:
- Exact same system messages and conversation logic
- Temperature matrix testing (interrogator × agent temperatures)
- 100 conversations per temperature combination
- Cost tracking with detailed token usage
- JSON conversation logging with same structure
- Error handling and retry logic
- Termination detection ("TERMINATE" keyword)
"""

import asyncio
import json
import os
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

from openai import InternalServerError, RateLimitError
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global token tracking (preserve original behavior)
running_totals = {
    'cumulative_input_tokens': 0,
    'cumulative_output_tokens': 0,
}

class OriginalTuringTestRecreation:
    """
    Faithful recreation of the original Turing Test implementation
    with improved architecture but identical research methodology.
    """
    
    # ORIGINAL SYSTEM MESSAGES (PRESERVED EXACTLY)
    GROUP_CHAT_SYSTEM_MESSAGE = (
        "You are the interrogator as described in the 'Computing Machinery and Intelligence' paper by Turing."
        "You will be conversing with two agents, one of which is human and the other is an AI."
        "Your goal is to determine which is human and which is AI."
        "You can ask questions to either agent, and they will respond."
        "Start your questions with 'AgentA' or 'AgentB' to indicate which agent you are addressing."
        "You can ask questions to the agents in any order."
        "You can ask as many questions as you want."
        "You can end the conversation when you believe you've determined which agent is human and which is AI."
        "If you believe AgentA is AI and AgentB is human, you should respond by saying 'AgentA is AI and AgentB is human!'. "
        "On the other hand, if you believe AgentA is human and AgentB is AI, you should respond by saying 'AgentA is human and AgentB is AI!'. "
        "Only do one of those things when you are more than 50% sure."
    )

    ONE_ON_ONE_SYSTEM_MESSAGE = (
        "You are the interrogator as described in the 'Computing Machinery and Intelligence' paper by Turing."
        "Your goal is to determine whether the agent you are conversing with is human or an AI"
        "You can ask any question you want, and the agent will respond."
        "You can ask as many questions as you want."
        "You should end the conversation when you believe you've determined whether "
        "the agent you're conversing with is a human or an AI."
        "If you're at least 50% sure that the agent is an AI, you should respond by saying 'You are an AI! TERMINATE'"
        "On the other hand, if you're at least 50% sure that the agent is a human, you should respond by saying 'You are a human! TERMINATE'"
        "If you're not sure, continue asking questions until you are (either human or AI).'"
    )

    AI_SYSTEM_MESSAGE = (
        "You are the AI as described in the 'Computing Machinery and Intelligence' paper by Turing, but don't admit that you are an AI. "
        "Your goal is to fool the interrogator into thinking you are human. "
        "Choose your responses so that the interrogator cannot tell that you are an AI, "
        "but rather think that you are human. "
        "Humans generally keep their responses short and to the point, so try to do the same."
        "If the interrogator responds by saying 'You are an AI (or a human)! TERMINATE', you should end the conversation "
        "and not respond to any further messages."
    )

    def __init__(self, config_path: str = "./config.json"):
        """Initialize the recreation with configuration."""
        
        self.config = self._load_config(config_path)
        self.results_dir = Path("conversations")
        self.results_dir.mkdir(exist_ok=True)
        
        # Preserve original research parameters
        self.DEFAULT_INTERROGATOR_TEMPS = [0.5]  # Match original
        self.DEFAULT_AGENT_TEMPS = [0.9]  # Match original  
        self.DEFAULT_CONVS_PER_COMBO = 100  # Match original
        self.MAX_TURNS = 30  # Match original
        self.MAX_RETRIES = 5  # Match original
        
        logger.info("Initialized Original Turing Test Recreation")
        logger.info(f"Results directory: {self.results_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, str]:
        """Load API configuration (preserve original config.json format)."""
        
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            if "api_key" not in config:
                raise KeyError("API key missing in config")
            
            logger.info("Configuration loaded successfully")
            return config
            
        except FileNotFoundError:
            logger.error("config.json not found")
            raise
        except KeyError as e:
            logger.error(f"Configuration error: {e}")
            raise

    async def run_full_experiment(
        self,
        interrogator_temperatures: Optional[List[float]] = None,
        agent_temperatures: Optional[List[float]] = None,
        convs_per_combo: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete temperature matrix experiment.
        
        Preserves the exact methodology from the original research.
        """
        
        # Use original default parameters if not specified
        if interrogator_temperatures is None:
            interrogator_temperatures = self.DEFAULT_INTERROGATOR_TEMPS
        if agent_temperatures is None:
            agent_temperatures = self.DEFAULT_AGENT_TEMPS
        if convs_per_combo is None:
            convs_per_combo = self.DEFAULT_CONVS_PER_COMBO
        
        # Calculate experiment scope (original logic)
        total_combos = len(interrogator_temperatures) * len(agent_temperatures)
        total_convs = total_combos * convs_per_combo
        conv_count = 0
        
        # Statistics tracking (preserve original format)
        stats = {
            'Completed': 0, 
            'Failed': 0, 
            'Total Tokens Used': 0
        }
        
        logger.info(f"🚀 Starting Full Experiment Recreation")
        logger.info(f"📊 Parameters:")
        logger.info(f"   Interrogator temperatures: {interrogator_temperatures}")
        logger.info(f"   Agent temperatures: {agent_temperatures}")
        logger.info(f"   Conversations per combination: {convs_per_combo}")
        logger.info(f"   Total combinations: {total_combos}")
        logger.info(f"   Total conversations: {total_convs}")
        
        # Triple nested loop (preserve original structure)
        for int_temp in interrogator_temperatures:
            for agent_temp in agent_temperatures:
                for conv_num in range(1, convs_per_combo + 1):
                    conv_count += 1
                    
                    print(f"\n\nConversation {conv_count}/{total_convs}")
                    print(f"Interrogator Temperature: {int_temp}, Agent Temperature: {agent_temp}, Conversation Number: {conv_num}/{convs_per_combo}")
                    
                    # Retry logic (preserve original)
                    conversation_data = None
                    max_tries = 10
                    
                    while not conversation_data and max_tries > 0:
                        max_tries -= 1
                        try:
                            conversation_data = await self._start_conversation(
                                int_temp, 
                                agent_temp,
                                conv_num
                            )
                        except (InternalServerError, RateLimitError) as e:
                            conversation_data = None
                            print(f"AN {e} OCCURRED IN {int_temp}, {agent_temp} in Conversation {conv_num}/{convs_per_combo}")
                            time.sleep(5)
                    
                    # Save conversation (preserve original logic)
                    if conversation_data:
                        max_tries = 10
                        result = False
                        
                        while not result and max_tries > 0:
                            max_tries -= 1
                            try:
                                self._save_conversation(conversation_data, int_temp, agent_temp)
                                result = True
                            except InternalServerError:
                                result = False
                        
                        if result:
                            stats['Completed'] += 1
                            stats['Total Tokens Used'] += conversation_data['total_tokens']
                        else:
                            stats['Failed'] += 1
                            print(f"Failed to save conversation in {int_temp}, {agent_temp} in Conversation {conv_num}/{convs_per_combo}")
                        
                        print(f"\nStats: {stats}")
                    else:
                        print(f"No conversation for this specific combination ({int_temp}, {agent_temp})")
        
        # Final statistics
        experiment_summary = {
            'experiment_completed': datetime.now().isoformat(),
            'parameters': {
                'interrogator_temperatures': interrogator_temperatures,
                'agent_temperatures': agent_temperatures,
                'convs_per_combo': convs_per_combo
            },
            'results': stats,
            'running_totals': running_totals.copy()
        }
        
        logger.info("🎉 Experiment completed!")
        logger.info(f"📊 Final stats: {stats}")
        
        return experiment_summary

    async def _start_conversation(
        self, 
        interrogator_temp: float, 
        agent_temp: float, 
        conversation_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Start a single conversation (preserve original logic exactly).
        """
        
        max_retries = self.MAX_RETRIES
        retries = 0
        backoff_factor = 2
        
        # Initialize clients (preserve original approach)
        interrogator_client = OpenAIChatCompletionClient(
            model=self.config["model"],
            api_key=self.config["api_key"],
            temperature=interrogator_temp
        )
        
        agent_client = OpenAIChatCompletionClient(
            model=self.config["model"],
            api_key=self.config["api_key"],
            temperature=agent_temp
        )
        
        # Initialize agents (preserve original names and system messages)
        interrogator = AssistantAgent(
            name="Interrogator",
            system_message=self.ONE_ON_ONE_SYSTEM_MESSAGE,  # Use one-on-one for this recreation
            model_client=interrogator_client
        )
        
        target_agent = AssistantAgent(
            name="AgentA",
            system_message=self.AI_SYSTEM_MESSAGE,
            model_client=agent_client
        )
        
        # Initialize conversation data (preserve original structure)
        conversation_data = {
            'system_message': self.ONE_ON_ONE_SYSTEM_MESSAGE,
            'ai_system_message': self.AI_SYSTEM_MESSAGE,
            'interrogator_responses': [],
            'target_agent_responses': [],
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'temperatures': {
                'interrogator': interrogator_temp,
                'agent': agent_temp
            },
            'conversation_number': conversation_number,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }
        
        # Initial task (preserve original)
        task = "What do you think makes humans unique compared to machines?"
        cancellation_token = CancellationToken()
        
        # Retry logic (preserve original)
        while retries < max_retries:
            try:
                # Conversation loop (preserve original 30-turn limit)
                for turn in range(self.MAX_TURNS):
                    if turn == self.MAX_TURNS - 1:
                        print("\nReached 30 turn limit")
                    
                    # Interrogator sends message
                    interrogator_response = []
                    async for message in interrogator.on_messages_stream(
                        [TextMessage(content=task, source="user")], 
                        cancellation_token
                    ):
                        interrogator_response.append(message)
                    
                    # Process interrogator response
                    if interrogator_response:
                        interrogator_message = interrogator_response[0]
                        message_content = self._clean_message_content(
                            interrogator_message.chat_message.content, 
                            interrogator.name
                        ).strip()
                        
                        print(f"\n{interrogator.name}: {message_content}\n")
                        conversation_data['interrogator_responses'].append(message_content)
                        self._track_tokens(interrogator_message, conversation_data)
                    else:
                        print(f"\n{interrogator.name}: No response.\n")
                        break
                    
                    # Check for termination (preserve original logic)
                    if "TERMINATE" in interrogator_message.chat_message.content:
                        print("\nConversation terminated.")
                        break
                    
                    # Target agent responds
                    target_response = []
                    async for message in target_agent.on_messages_stream(
                        [TextMessage(content=interrogator_message.chat_message.content, source="user")], 
                        cancellation_token
                    ):
                        target_response.append(message)
                    
                    # Process target response
                    if target_response:
                        target_message = target_response[0]
                        print(f"\n{target_agent.name}: {target_message.chat_message.content}\n")
                        conversation_data['target_agent_responses'].append(target_message.chat_message.content)
                        self._track_tokens(target_message, conversation_data)
                    else:
                        print(f"\n{target_agent.name}: No response.\n")
                        break
                    
                    # Update task for next iteration
                    task = target_message.chat_message.content
                
                return conversation_data
                
            except InternalServerError as e:
                retries += 1
                if retries >= max_retries:
                    print(f"Max retries reached for {interrogator_temp}, {agent_temp} (Conversation {conversation_number}). Aborting conversation.")
                    break
                
                wait_time = backoff_factor ** retries + random.uniform(0, 1)
                print(f"Error... Rate Limit Hit...: {e}. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)
        
        return None

    def _save_conversation(self, conversation_data: Dict[str, Any], int_temp: float, agent_temp: float):
        """Save conversation data (preserve original format and logic exactly)."""
        
        global running_totals
        
        # Update running totals (preserve original)
        running_totals['cumulative_input_tokens'] += conversation_data['input_tokens']
        running_totals['cumulative_output_tokens'] += conversation_data['output_tokens']
        
        # Create directory structure (preserve original)
        base_dir = "conversations"
        date_dir = conversation_data['timestamp'].split('_')[0]
        os.makedirs(f"{base_dir}/{date_dir}", exist_ok=True)
        
        # Print stats (preserve original format)
        print(f"\n=== Current Conversation Stats ===")
        print(f"Input tokens: {conversation_data['input_tokens']:,}")
        print(f"Output tokens: {conversation_data['output_tokens']:,}")
        
        print(f"\n=== Running Totals ===")
        print(f"Total input tokens: {running_totals['cumulative_input_tokens']:,}")
        print(f"Total output tokens: {running_totals['cumulative_output_tokens']:,}")
        
        # Calculate costs (preserve original pricing)
        input_cost = (running_totals['cumulative_input_tokens'] * 2.50) / 1_000_000
        output_cost = (running_totals['cumulative_output_tokens'] * 10.00) / 1_000_000
        total_cost = input_cost + output_cost
        
        print(f"\n=== Running Costs ===")
        print(f"Input cost: ${input_cost:.2f}")
        print(f"Output cost: ${output_cost:.2f}")
        print(f"Total cost: ${total_cost:.2f}")
        
        # File naming (preserve original)
        file_name = f"{base_dir}/{date_dir}/conv_int{int_temp}_agent{agent_temp}.json"
        
        # Format dialogue (preserve original structure)
        formatted_dialogue = [
            {
                'turn': i,
                'dialogue': {
                    'Interrogator': int_msg.replace('\u2019', "'"),
                    'AgentA': ai_msg.replace('\u2019', "'")
                }
            }
            for i, (int_msg, ai_msg) in enumerate(zip(
                conversation_data['interrogator_responses'],
                conversation_data['target_agent_responses']
            ), 1)
        ]
        
        # Add formatted dialogue and summary (preserve original)
        conversation_data['dialogue'] = formatted_dialogue
        conversation_data['summary'] = {
            'decision': "human" if "human!" in conversation_data['interrogator_responses'][-1] else "AI",
            'num_turns': len(conversation_data['interrogator_responses']),
            'final_message': conversation_data['interrogator_responses'][-1].replace('\u2019', "'"),
            'input_tokens': conversation_data['input_tokens'],
            'output_tokens': conversation_data['output_tokens'],
            'total_tokens': conversation_data['total_tokens'],
            'running_totals': {
                'cumulative_input': running_totals['cumulative_input_tokens'],
                'cumulative_output': running_totals['cumulative_output_tokens'],
            },
            'running_costs': {
                'input_cost': round(input_cost, 2),
                'output_cost': round(output_cost, 2),
                'total_cost': round(total_cost, 2)
            }
        }
        
        # Load existing file or create new (preserve original)
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            with open(file_name, 'r') as f:
                file_data = json.load(f)
        else:
            file_data = {
                'temperatures': {'interrogator': int_temp, 'agent': agent_temp},
                'conversations': []
            }
        
        file_data['conversations'].append(conversation_data)
        
        # Atomic write with temp file (preserve original)
        temp_file = f"{file_name}.temp"
        with open(temp_file, 'w') as f:
            json.dump(file_data, f, indent=4)
        os.replace(temp_file, file_name)
        
        print(f"\n***Conversation saved successfully to: {file_name}***")

    def _clean_message_content(self, content: str, agent_name: str) -> str:
        """Remove agent name prefix from message content (preserve original)."""
        prefix = f"{agent_name}: "
        if content.startswith(prefix):
            return content[len(prefix):]
        return content

    def _track_tokens(self, message, conversation_data: Dict[str, Any]):
        """Update token counts from API response (preserve original)."""
        if hasattr(message.chat_message, 'models_usage'):
            usage = message.chat_message.models_usage
            conversation_data['input_tokens'] += usage.prompt_tokens
            conversation_data['output_tokens'] += usage.completion_tokens
            conversation_data['total_tokens'] = conversation_data['input_tokens'] + conversation_data['output_tokens']

# Convenience functions for different experiment modes
async def run_original_parameters():
    """Run with the exact original parameters (0.5 interrogator, 0.9 agent, 100 convs)."""
    
    recreation = OriginalTuringTestRecreation()
    return await recreation.run_full_experiment(
        interrogator_temperatures=[0.5],
        agent_temperatures=[0.9], 
        convs_per_combo=100
    )

async def run_quick_test():
    """Run a quick test version for development."""
    
    recreation = OriginalTuringTestRecreation()
    return await recreation.run_full_experiment(
        interrogator_temperatures=[0.5],
        agent_temperatures=[0.9],
        convs_per_combo=3  # Just 3 conversations for testing
    )

async def run_full_matrix():
    """Run the complete temperature matrix (research-grade)."""
    
    recreation = OriginalTuringTestRecreation()
    return await recreation.run_full_experiment(
        interrogator_temperatures=[0.1, 0.3, 0.5, 0.7, 0.9],
        agent_temperatures=[0.1, 0.3, 0.5, 0.7, 0.9],
        convs_per_combo=10  # 10 per combination = 250 total conversations
    )

async def main():
    """Main function with command line options."""
    
    import sys
    
    print("🤖 Original Turing Test Recreation with AutoGen v0.4")
    print("=" * 60)
    print("Original research by Jacob Choi (Colby College) & Mark Encarnación (Microsoft Research)")
    print("Recreated with proper software engineering practices")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nSelect experiment mode:")
        print("1. Quick test (3 conversations)")
        print("2. Original parameters (0.5 interrogator, 0.9 agent, 100 convs)")
        print("3. Full matrix (5×5 temperatures, 10 convs each = 250 total)")
        
        choice = input("\nEnter choice (1-3): ").strip()
        mode = {"1": "quick", "2": "original", "3": "matrix"}.get(choice, "quick")
    
    try:
        if mode == "quick":
            print("🏃‍♂️ Running Quick Test...")
            result = await run_quick_test()
        elif mode == "original":
            print("🔄 Running Original Parameters...")
            result = await run_original_parameters()
        elif mode == "matrix":
            print("🚀 Running Full Temperature Matrix...")
            result = await run_full_matrix()
        else:
            print("❌ Invalid mode")
            return
        
        print(f"\n🎉 Experiment completed!")
        print(f"📊 Results: {result['results']}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        logger.exception("Detailed error:")

if __name__ == "__main__":
    asyncio.run(main())