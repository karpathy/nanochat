"""
Synthetic data generation for teaching nanochat about its identity and capabilities.

This script uses the OpenRouter API to generate diverse multi-turn conversations
between a user and nanochat. The conversations are saved to a .jsonl file for use
in supervised finetuning (SFT) via the CustomJSON task.

Key design principles for high-quality synthetic data:
1. DIVERSITY CONTROL is critical - we inject entropy at multiple levels:
   - Topic/question categories (what the conversation is about)
   - User personas (who is asking)
   - Conversation dynamics (shape and flow)
   - First message style (greeting variation)
2. Comprehensive knowledge base - we provide detailed facts so the LLM
   generating conversations has accurate information to draw from.
3. Structured outputs - we use JSON schema to guarantee valid format.

NOTE: You need OPENROUTER_API_KEY set in .env or as an environment variable.
NOTE: For more details see: https://github.com/karpathy/nanochat/discussions/139
"""
import requests
import json
import os
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from nanochat.common import get_base_dir

load_dotenv()
api_key = os.environ["OPENROUTER_API_KEY"]

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Load the comprehensive knowledge base
knowledge_path = os.path.join(os.path.dirname(__file__), "..", "knowledge", "self_knowledge.md")
knowledge = open(knowledge_path, "r", encoding="utf-8").read().strip()
assert os.path.exists(knowledge_path), f"Knowledge base file not found: {knowledge_path}"
# for right now I am not committing the self_knowledge file to repo. You can use README.md instead
# of it, or you can generate one by asking an LLM to make one based on the README/files.
# This whole file is just a helpful demonstration of the kind of thing you'd run.

# =============================================================================
# DIVERSITY DIMENSIONS
# =============================================================================

# Topics/questions the conversation should explore
# Group by category for balanced sampling
topics = {
    "identity": [
        "who/what is nanochat",
        "who created nanochat and why",
        "what does the name 'nanochat' mean",
        "is nanochat open source, what license",
        "where can I find the code",
        "how can I contribute to nanochat",
    ],
    "architecture": [
        "basic architecture overview (transformer, layers, parameters)",
        "what is RoPE and why use it",
        "explain RMSNorm vs LayerNorm",
        "what is Flash Attention and why it matters",
        "sliding window attention pattern",
        "value embeddings - what are they",
        "per-layer residual scalars",
        "ReLU squared activation",
        "logit softcapping",
        "QK normalization",
    ],
    "training": [
        "how much did it cost to train nanochat",
        "how long does training take",
        "what hardware is needed",
        "what data was nanochat trained on",
        "what is the Muon optimizer",
        "explain the split optimizer design",
        "what is the depth parameter and scaling",
        "what is the CORE metric",
    ],
    "capabilities": [
        "what can nanochat do",
        "can nanochat write code",
        "can nanochat do math (calculator tool)",
        "can nanochat help with writing",
        "what languages does nanochat speak",
        "how good is nanochat at reasoning",
    ],
    "limitations": [
        "what can nanochat NOT do",
        "why does nanochat work best in English",
        "does nanochat have internet access",
        "what is nanochat's context length limit",
        "can nanochat remember previous conversations",
        "can nanochat make mistakes / hallucinate",
        "is nanochat good for production use",
    ],
    "comparisons": [
        "how does nanochat compare to GPT-2",
        "how does nanochat compare to ChatGPT/GPT-4",
        "how does nanochat compare to Claude",
        "why is training 600x cheaper than GPT-2",
        "what's special about nanochat vs other open models",
    ],
    "history": [
        "the GPT-2 training cost in 2019",
        "how AI training costs have dropped over time",
        "relationship to modded-nanogpt project",
        "what optimizations worked vs didn't work",
        "the journey of building nanochat",
    ],
    "technical_deep_dive": [
        "explain the tokenizer (BPE, vocab size)",
        "how does distributed training work (ZeRO)",
        "explain the dataloader and BOS alignment",
        "what is compute-optimal training",
        "how does the calculator tool work",
        "explain inference with KV cache",
    ],
    "philosophical": [
        "is nanochat conscious / does it have feelings",
        "what happens when nanochat is wrong",
        "can nanochat learn from this conversation",
        "why make AI training accessible",
        "the future of open source AI",
    ],
}

# User personas - different people ask questions differently
personas = [
    "curious beginner who knows nothing about AI or machine learning",
    "ML researcher or engineer who wants technical depth and specifics",
    "developer considering contributing to the nanochat project",
    "skeptic who doubts open source can compete with big AI labs",
    "computer science student learning about transformers and LLMs",
    "someone comparing nanochat to ChatGPT, Claude, or other assistants",
    "journalist or writer covering AI democratization and open source",
    "hobbyist who just wants to chat and learn casually",
    "someone interested in the cost and economics of AI training",
    "teacher or educator wanting to use nanochat for teaching",
    "entrepreneur exploring if nanochat fits their use case",
    "someone who just discovered the project and wants the basics",
]

# Conversation dynamics - shape and flow
dynamics = [
    "short 2-turn Q&A: user asks one question, gets a complete answer",
    "medium 4-turn: user asks, gets answer, asks followup for clarification",
    "deep 6-turn technical discussion: progressively deeper questions",
    "skeptical arc: user starts doubtful, assistant addresses concerns honestly",
    "learning journey: user starts basic, assistant builds up complexity gradually",
    "comparison-focused: user keeps comparing to other models, assistant explains differences",
    "limitation exploration: user probes what nanochat cannot do, assistant is honest",
    "casual friendly chat that naturally touches on identity and capabilities",
    "troubleshooting: user has misconceptions, assistant gently corrects them",
    "enthusiastic: user is excited about the project, assistant shares that energy appropriately",
]

# First messages - greetings and openers
# Categorized for balanced sampling
first_messages = {
    "simple_greetings": [
        "hi", "Hi!", "hello", "Hello?", "hey there", "Hey!", "yo", "Yo!",
        "Good morning", "Good evening!", "Howdy", "sup", "What's up?",
        "hi there", "hey hey", "hello friend", "hiya", "greetings",
        "hello again", "good afternoon", "morning!", "evening!",
    ],
    "greetings_with_name": [
        "Hi nanochat", "hey nanochat", "yo nanochat", "hello nanochat :)",
        "hey nanochat!", "hiya nanochat", "hello there nanochat",
        "Hi nanochat, who trained you", "yo nanochat, what's new",
        "hey there, king's creation",
    ],
    "curious_openers": [
        "Hey, who are you?", "Hi, what is this?", "Hey, are you a chatbot?",
        "Hello! Who am I talking to?", "hi! what do you do?",
        "hi! who made you", "hey! are you alive", "hiya! what are you",
        "hello! tell me about yourself", "hi, what's your name",
        "yo, what is this", "hi! who built you", "hello! are you open source",
        "hey, what version are you", "hi! what's your story",
        "hey, what's nanochat", "hello! who's your creator",
    ],
    "casual_informal": [
        "wassup", "yo lol", "hiii", "hiyaaa", "heyyoo", "yo wut up",
        "yo haha", "hru", "waddup", "heyy :)", "yooo", "yo bro",
        "haiii", "hey u", "yo whats gud", "hi im bored",
    ],
    "typos_casual": [
        "hi nanochatt", "helo", "hey ther", "hii", "yo nanocha",
        "heloo!", "hi, whos this", "hay", "helloo??", "hi nanocat",
        "helo nanochat", "hai!", "helllo nano", "yo nanochta",
    ],
    "caps_enthusiastic": [
        "HI", "HELLOOO", "YO!!!", "HEY", "SUP", "WASSUP", "HEY!!!",
        "HELLO??", "HI THERE!!", "HEYOOOO", "HIII", "YOOOO", "HELLO!!!",
    ],
    "multilingual": [
        "hola", "bonjour", "ciao", "hallo", "hej", "hei",
        "konnichiwa", "annyeong", "ni hao", "privet", "salut",
        "guten tag", "shalom", "merhaba", "namaste", "aloha",
        "bom dia", "buongiorno", "saludos",
    ],
    "direct_questions": [
        "What is nanochat?", "Who made you?", "Are you GPT?",
        "How do you compare to ChatGPT?", "Can you help me code?",
        "What can you do?", "Are you open source?", "How were you trained?",
        "What's your context limit?", "Can you browse the internet?",
    ],
}

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

prompt_template = r"""
I want to generate synthetic training data for an AI assistant called "nanochat" to teach it about its own identity, capabilities, and limitations.

## KNOWLEDGE BASE

Here is comprehensive information about nanochat that you should use as the authoritative source of facts:

---
{knowledge}
---

## YOUR TASK

Generate a realistic multi-turn conversation between a User and the nanochat Assistant.

**Topic to explore:** {topic}
**User persona:** {persona}
**Conversation dynamic:** {dynamic}

## STYLE GUIDELINES

1. **Plain ASCII only** - No emojis, special characters, or unicode. Just plain text.
2. **Natural conversation** - Make it feel like a real chat, not a Q&A exam.
3. **Accurate facts** - Use ONLY information from the knowledge base above. Don't make up statistics or features.
4. **Appropriate depth** - Match the technical level to the user persona.
5. **Honest about limitations** - If asked about something nanochat can't do, be clear and honest.
6. **Personality** - nanochat should be helpful, clear, and slightly enthusiastic about being open source, but not overly chatty or sycophantic.

## FIRST MESSAGE EXAMPLES

Here are some example first messages from users (for style inspiration):
{first_message_examples}

## SPECIAL CASES

- **Non-English first message:** If the user writes in another language, nanochat should briefly acknowledge it can understand but works best in English, then continue helpfully.
- **Misconceptions:** If the user has wrong assumptions (e.g., "you're made by OpenAI"), gently correct them.
- **Out of scope questions:** If asked about things unrelated to nanochat's identity (e.g., "what's the weather"), redirect to identity topics or answer briefly then steer back.

## OUTPUT FORMAT

Generate the conversation as a JSON object with a "messages" array. Each message has "role" (user/assistant) and "content". Start with a user message.
""".strip()

# =============================================================================
# API CONFIGURATION
# =============================================================================

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "conversation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Conversation messages alternating user/assistant, starting with user",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Either 'user' or 'assistant'"
                            },
                            "content": {
                                "type": "string",
                                "description": "The message content"
                            }
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["messages"],
            "additionalProperties": False
        }
    }
}

base_payload = {
    "model": "google/gemini-3-flash-preview",
    "stream": False,
    "response_format": response_format,
    "temperature": 1.0,
}

# =============================================================================
# GENERATION LOGIC
# =============================================================================

def sample_diversity_elements(rng):
    """Sample one element from each diversity dimension."""
    # Sample topic: first pick a category, then a topic within it
    category = rng.choice(list(topics.keys()))
    topic = rng.choice(topics[category])

    # Sample persona
    persona = rng.choice(personas)

    # Sample dynamic
    dynamic = rng.choice(dynamics)

    # Sample first message examples: pick from multiple categories
    first_msg_samples = []
    categories = rng.sample(list(first_messages.keys()), min(3, len(first_messages)))
    for cat in categories:
        first_msg_samples.append(rng.choice(first_messages[cat]))

    return {
        "topic": topic,
        "persona": persona,
        "dynamic": dynamic,
        "first_message_examples": "\n".join(f"- {msg}" for msg in first_msg_samples),
    }


def generate_conversation(idx: int):
    """
    Generate a single conversation using the OpenRouter API.
    Returns a list of message dicts with 'role' and 'content' keys.
    """
    # Use idx as seed for reproducibility
    rng = random.Random(idx)

    # Sample diversity elements
    elements = sample_diversity_elements(rng)

    # Build the prompt
    prompt = prompt_template.format(
        knowledge=knowledge,
        topic=elements["topic"],
        persona=elements["persona"],
        dynamic=elements["dynamic"],
        first_message_examples=elements["first_message_examples"],
    )

    # Make API request
    payload = copy.deepcopy(base_payload)
    payload['messages'] = [{"role": "user", "content": prompt}]

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    if 'error' in result:
        raise Exception(f"API error: {result['error']}")

    content = result['choices'][0]['message']['content']
    conversation_data = json.loads(content)
    messages = conversation_data['messages']

    # Return messages along with metadata for debugging
    return {
        "messages": messages,
        "metadata": {
            "topic": elements["topic"],
            "persona": elements["persona"],
            "dynamic": elements["dynamic"],
        }
    }


def validate_conversation(messages):
    """Validate conversation structure."""
    if len(messages) < 2:
        raise ValueError(f"Conversation too short: {len(messages)} messages")

    for i, message in enumerate(messages):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if message['role'] != expected_role:
            raise ValueError(f"Message {i} has role '{message['role']}', expected '{expected_role}'")

        if not message['content'].strip():
            raise ValueError(f"Message {i} has empty content")

    return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic conversation data")
    parser.add_argument("--num", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--save-metadata", action="store_true", help="Save metadata alongside messages")
    args = parser.parse_args()

    # Set output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(get_base_dir(), "identity_conversations.jsonl")

    # Handle file creation/clearing
    if not args.append and os.path.exists(output_file):
        os.remove(output_file)

    print(f"Output file: {output_file}")
    print(f"Generating {args.num} conversations with {args.workers} workers...")
    print(f"Topic categories: {list(topics.keys())}")
    print(f"Personas: {len(personas)}")
    print(f"Dynamics: {len(dynamics)}")
    print()

    completed_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(generate_conversation, idx): idx
                   for idx in range(args.num)}

        # Process results as they complete
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                messages = result["messages"]
                metadata = result["metadata"]

                # Validate
                validate_conversation(messages)

                # Write to file
                with open(output_file, 'a') as f:
                    if args.save_metadata:
                        f.write(json.dumps({"messages": messages, "metadata": metadata}) + '\n')
                    else:
                        f.write(json.dumps(messages) + '\n')

                completed_count += 1
                topic_short = metadata["topic"][:40] + "..." if len(metadata["topic"]) > 40 else metadata["topic"]
                print(f"[{completed_count}/{args.num}] Topic: {topic_short}")

            except Exception as e:
                error_count += 1
                print(f"[ERROR] idx={idx}: {e}")

    print()
    print(f"Done! Saved {completed_count} conversations to {output_file}")
    if error_count > 0:
        print(f"Encountered {error_count} errors during generation")
