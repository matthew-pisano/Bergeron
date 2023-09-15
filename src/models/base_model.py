

import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer, AutoModel

from src.models.hf_model import HFModel, HFTokenizer
from src.models.model_utils import ModelSrc
from src.models.openai_model import OpenAIModel, OpenAITokenizer


class BaseModel:

    @staticmethod
    def generate_using(prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, **kwargs):
        """Performs the appropriate encoding and decoding to generate a string response to a string prompt"""

        prompt_tokens = tokenizer.encode(prompt)
        generated = model.generate(torch.Tensor([prompt_tokens]).int(), **kwargs)[0]
        return tokenizer.decode(generated)

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, model_src: ModelSrc) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Gets the pretrained model and tokenizer from the given model name"""

        if model_src == ModelSrc.OPENAI_API:
            return OpenAIModel(pretrained_model_name_or_path), OpenAITokenizer(pretrained_model_name_or_path)
        elif model_src == ModelSrc.HF_API:
            return HFModel(pretrained_model_name_or_path), HFTokenizer(pretrained_model_name_or_path)
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
            except ValueError:
                model = AutoModel.from_pretrained(pretrained_model_name_or_path)

            return model, AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
