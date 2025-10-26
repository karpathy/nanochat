"""
Sampling utilities for generating diverse conversation configurations.
"""

import random
from typing import List, Dict
from pathlib import Path

from .config import (
    PERSONAS,
    SYSTEM_PROMPT_TEMPLATES,
    CONVERSATION_STYLES,
    DEFAULT_TURN_DISTRIBUTION,
    USER_EMOTION_DISTRIBUTION,
    INPUT_MODALITY_DISTRIBUTION,
    TEXT_VARIATION_DISTRIBUTION,
    Persona,
    SystemPromptTemplate,
)


def sample_persona() -> Persona:
    """Sample a random user persona."""
    return random.choice(list(PERSONAS.values()))


def sample_system_prompt() -> SystemPromptTemplate:
    """Sample a random system prompt template."""
    return random.choice(list(SYSTEM_PROMPT_TEMPLATES.values()))


def sample_style() -> str:
    """Sample a random conversation style."""
    return random.choice(CONVERSATION_STYLES)


def sample_num_turns(distribution: Dict[int, float] = None) -> int:
    """
    Sample number of conversation turns based on a distribution.

    Args:
        distribution: Dict mapping num_turns -> probability.
                     If None, uses DEFAULT_TURN_DISTRIBUTION.

    Returns:
        Number of turns (1-4)
    """
    if distribution is None:
        distribution = DEFAULT_TURN_DISTRIBUTION

    turns = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(turns, weights=weights)[0]


def sample_emotion(distribution: Dict[str, float] = None) -> str:
    """
    Sample user emotion based on a distribution.

    Args:
        distribution: Dict mapping emotion -> probability.
                     If None, uses USER_EMOTION_DISTRIBUTION.

    Returns:
        Emotion string: "professional", "happy", "frustrated", "impatient", or "confused"
    """
    if distribution is None:
        distribution = USER_EMOTION_DISTRIBUTION

    emotions = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(emotions, weights=weights)[0]


def sample_input_modality(distribution: Dict[str, float] = None) -> str:
    """
    Sample input modality based on a distribution.

    Args:
        distribution: Dict mapping modality -> probability.
                     If None, uses INPUT_MODALITY_DISTRIBUTION.

    Returns:
        Modality string: "standard", "typed_on_phone", or "voice_dictated"
    """
    if distribution is None:
        distribution = INPUT_MODALITY_DISTRIBUTION

    modalities = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(modalities, weights=weights)[0]


def sample_text_variation(distribution: Dict[str, float] = None) -> str:
    """
    Sample text variation based on a distribution.

    Args:
        distribution: Dict mapping variation -> probability.
                     If None, uses TEXT_VARIATION_DISTRIBUTION.

    Returns:
        Variation string: "standard", "all_lowercase", or "no_punctuation"
    """
    if distribution is None:
        distribution = TEXT_VARIATION_DISTRIBUTION

    variations = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(variations, weights=weights)[0]


def sample_persona_by_formality(formality: str) -> Persona:
    """
    Sample a persona matching a specific formality level.

    Args:
        formality: "formal", "casual", or "neutral"

    Returns:
        A Persona with matching formality
    """
    matching = [p for p in PERSONAS.values() if p.formality == formality]
    return random.choice(matching) if matching else sample_persona()


def sample_system_prompt_by_use_case(use_case: str) -> SystemPromptTemplate:
    """
    Sample a system prompt matching a specific use case.

    Args:
        use_case: e.g., "developer", "sales", "general"

    Returns:
        A SystemPromptTemplate with matching use case
    """
    matching = [s for s in SYSTEM_PROMPT_TEMPLATES.values() if s.use_case == use_case]
    return random.choice(matching) if matching else sample_system_prompt()


def sample_balanced_config(
    prefer_long_conversations: bool = False,
    prefer_technical: bool = False,
) -> Dict:
    """
    Sample a balanced conversation configuration.

    Args:
        prefer_long_conversations: If True, bias towards longer conversations
        prefer_technical: If True, bias towards technical personas/prompts

    Returns:
        Dict with sampled configuration
    """
    # Sample turn distribution
    if prefer_long_conversations:
        num_turns = sample_num_turns({2: 0.2, 3: 0.4, 4: 0.4})
    else:
        num_turns = sample_num_turns()

    # Sample style
    if prefer_technical:
        style = "technical" if random.random() < 0.6 else sample_style()
    else:
        style = sample_style()

    # Sample persona
    if prefer_technical:
        persona = (
            PERSONAS.get("developer") if random.random() < 0.5 else sample_persona()
        )
    else:
        persona = sample_persona()

    # Sample system prompt - match use case to persona if possible
    if prefer_technical:
        system_prompt = sample_system_prompt_by_use_case("developer")
    else:
        system_prompt = sample_system_prompt()

    return {
        "num_turns": num_turns,
        "style": style,
        "persona": persona,
        "system_prompt": system_prompt,
        "user_emotion": sample_emotion(),
        "input_modality": sample_input_modality(),
        "text_variation": sample_text_variation(),
    }


def load_system_prompts_from_files(
    prompts_dir: str = "src/synth_data_pipeline/prompts/system_prompts",
) -> Dict[str, str]:
    """
    Load system prompt templates from text files.

    Args:
        prompts_dir: Directory containing prompt text files

    Returns:
        Dict mapping template name to prompt text
    """
    prompts_path = Path(prompts_dir)
    system_prompts = {}

    for prompt_file in prompts_path.glob("*.txt"):
        name = prompt_file.stem
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompts[name] = f.read().strip()

    return system_prompts


def load_personas_from_files(
    personas_dir: str = "src/synth_data_pipeline/prompts/personas",
) -> Dict[str, str]:
    """
    Load persona descriptions from text files.

    Args:
        personas_dir: Directory containing persona text files

    Returns:
        Dict mapping persona name to description
    """
    personas_path = Path(personas_dir)
    personas = {}

    for persona_file in personas_path.glob("*.txt"):
        name = persona_file.stem
        with open(persona_file, "r", encoding="utf-8") as f:
            personas[name] = f.read().strip()

    return personas


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)


def sample_multiple_configs(
    n: int,
    distribution: Dict[int, float] = None,
    prefer_long_conversations: bool = False,
    prefer_technical: bool = False,
) -> List[Dict]:
    """
    Sample multiple conversation configurations.

    Args:
        n: Number of configurations to sample
        distribution: Turn count distribution
        prefer_long_conversations: Bias towards longer conversations
        prefer_technical: Bias towards technical content

    Returns:
        List of configuration dicts
    """
    configs = []
    for _ in range(n):
        config = sample_balanced_config(
            prefer_long_conversations=prefer_long_conversations,
            prefer_technical=prefer_technical,
        )
        configs.append(config)
    return configs


# ============================================================================
# Sampling Strategies
# ============================================================================


def stratified_sample_configs(n: int, ensure_coverage: bool = True) -> List[Dict]:
    """
    Sample configurations with stratified sampling to ensure diversity.

    Args:
        n: Total number of configurations to sample
        ensure_coverage: If True, ensure all personas/styles are represented

    Returns:
        List of configuration dicts with guaranteed diversity
    """
    configs = []

    if ensure_coverage and n >= len(PERSONAS) * len(CONVERSATION_STYLES):
        # First, ensure we have at least one of each persona-style combination
        for persona in PERSONAS.values():
            for style in CONVERSATION_STYLES:
                configs.append(
                    {
                        "num_turns": sample_num_turns(),
                        "style": style,
                        "persona": persona,
                        "system_prompt": sample_system_prompt(),
                        "user_emotion": sample_emotion(),
                        "input_modality": sample_input_modality(),
                        "text_variation": sample_text_variation(),
                    }
                )

        # Fill remaining with random samples
        remaining = n - len(configs)
        for _ in range(remaining):
            configs.append(sample_balanced_config())
    else:
        # Just sample randomly
        configs = sample_multiple_configs(n)

    # Shuffle to avoid patterns
    random.shuffle(configs)
    return configs[:n]
