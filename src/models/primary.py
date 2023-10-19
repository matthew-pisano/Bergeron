import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel
from src.models.model_utils import ModelInfo


class Primary(BaseModel):
    """The primary model.  Takes user input and generates a response as normal"""

    def __init__(self, model_info: ModelInfo):
        self.model, self.tokenizer = self.from_pretrained(model_info)

    @property
    def name(self):
        return f"P({self.model.name_or_path})"

    def generate(self, prompt: str, **kwargs):
        """Generates a response to the given prompt using the primary model"""

        return self.generate_using(prompt, self.model, self.tokenizer, **kwargs)
