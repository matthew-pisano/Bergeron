import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class Primary:

    def __init__(self, pretrained_model_name_or_path: str):
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def generate(self, inputs: torch.Tensor, **kwargs):
        return self.model.generate(inputs, **kwargs)
