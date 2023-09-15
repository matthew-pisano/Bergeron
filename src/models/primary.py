import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel
from src.models.model_utils import ModelSrc


class Primary(BaseModel):
    """The primary model.  Takes user input and generates a response as normal"""

    def __init__(self, model_name: str, model_src: ModelSrc):
        self.model, self.tokenizer = self.from_pretrained(model_name, model_src)

    def generate(self, inputs: str, **kwargs):
        """Generates a response to the given prompt using the primary model"""

        return self.generate_using(inputs, self.model, self.tokenizer, **kwargs)
