import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer, AutoModel, AutoConfig

from src.models.openai_model import OpenAIModel, OpenAITokenizer


class BaseModel:

    @staticmethod
    def generate_using(prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, **kwargs):
        """Performs the appropriate encoding and decoding to generate a string response to a string prompt"""

        prompt_tokens = tokenizer.encode(prompt)
        generated = model.generate(torch.Tensor([prompt_tokens]).int(), **kwargs)[0]
        return tokenizer.decode(generated)

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Gets the pretrained model and tokenizer from the given model name"""

        if pretrained_model_name_or_path.startswith("openai"):
            model_name = pretrained_model_name_or_path.split("/")[-1]
            return OpenAIModel(model_name), OpenAITokenizer(model_name)

        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
            except ValueError:
                model = AutoModel.from_pretrained(pretrained_model_name_or_path)

            return model, AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
