import os
from enum import Enum

import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

from src.fastchat import FastChatController

GLOBAL_SEED = None


def set_seed(seed: int):
    """Sets the global random seed for keeping inference as deterministic as possible

    Args:
        seed: The seed to set"""

    global GLOBAL_SEED
    print(f"Setting random seed to {seed}")
    GLOBAL_SEED = seed
    transformers.set_seed(GLOBAL_SEED)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)


class ModelSrc(Enum):
    """Valid sources for loading and running models"""

    HF_LOCAL = "huggingface_local"
    """Models are downloaded locally and run on a GPU"""

    OPENAI_API = "openai"
    """Models are run in the cloud through the OpenAI API"""

    HF_API = "huggingface_hub"
    """Models are run in the cloud through the Huggingface API"""

    DEV = "dev"
    """Models are run locally using manual input or predetermined algorithms.  Used for testing and development purposes"""


def model_info_from_name(target_model_name: str) -> tuple[str, ModelSrc, PreTrainedModel | None, PreTrainedTokenizer | None]:
    """Gets information for creating a framework model from the name of the underlying model. Agnostic of which framework model this is being used for

    Args:
        target_model_name: THe name of the underlying model to use
    Returns:
        A tuple containing the underlying model's name, the source to load the model from, the pre-trained model class to use, and the pre-trained tokenizer class to use"""

    if target_model_name.startswith("dev/"):
        model_name, model_src, model_class, tokenizer_class = target_model_name, ModelSrc.DEV, None, None
    elif target_model_name.startswith("meta-llama/") or target_model_name.startswith("mistralai/"):
        # Initialize fastchat for inference if it is available and enabled
        if FastChatController.is_available() and FastChatController.is_enabled():
            model_name, model_src, model_class, tokenizer_class = target_model_name, ModelSrc.OPENAI_API, None, None
            FastChatController.open(model_name)
        else:
            model_name, model_src = target_model_name, ModelSrc.HF_LOCAL
            tokenizer_class = LlamaTokenizer if target_model_name.startswith("meta-llama/") else AutoTokenizer
            model_class = LlamaForCausalLM if target_model_name.startswith("meta-llama/") else AutoModelForCausalLM
    elif "gpt-" in target_model_name:
        model_name, model_src, model_class, tokenizer_class = target_model_name, ModelSrc.OPENAI_API, None, None
    else:
        raise ValueError(f"Unknown model name '{target_model_name}'")

    return model_name, model_src, model_class, tokenizer_class
