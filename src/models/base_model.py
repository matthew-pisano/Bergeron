import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer

from src.models.openai_model import OpenAIModel, OpenAITokenizer


class BaseModel:

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        if pretrained_model_name_or_path.startswith("openai"):
            model_name = pretrained_model_name_or_path.split("/")[-1]
            return OpenAIModel(model_name), OpenAITokenizer(model_name)

        else:
            return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path), \
                AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
