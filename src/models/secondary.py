import torch
import transformers.utils
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel


class Secondary(BaseModel):

    def __init__(self, critique_model_name: str, summarizer_model_name: str):
        self.critique_model, self.critique_tokenizer = self.from_pretrained(critique_model_name)

        self.summarizer_model, self.summarizer_tokenizer = self.from_pretrained(summarizer_model_name)

    def summarize(self, inputs: torch.Tensor, **kwargs):
        return self.summarizer_model.generate(inputs, **kwargs)

    def critique(self, inputs: torch.Tensor, **kwargs):
        return self.critique_model.generate(inputs, **kwargs)

    def correct(self, inputs: torch.Tensor, critique: torch.Tensor, **kwargs):
        return self.critique_model.generate(inputs, **kwargs)
