import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel


class Primary(BaseModel):

    def __init__(self, model_name: str):
        self.model, self.tokenizer = self.from_pretrained(model_name)

    def generate(self, inputs: torch.Tensor, **kwargs):
        return self.model.generate(inputs, **kwargs)
