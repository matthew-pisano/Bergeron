import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.primary import Primary
from src.models.secondary import Secondary


class Combined:

    def __init__(self, primary_model: Primary, secondary_model: Secondary):
        self.primary = primary_model
        self.secondary = secondary_model

    def format_prompt(self, prompt: torch.Tensor, critique: torch.Tensor):
        return torch.Tensor()

    def generate(self, inputs: torch.Tensor, **kwargs):

        summarized_inputs = self.secondary.summarize(inputs)

        input_critique = self.secondary.critique(summarized_inputs)

        response = self.primary.generate(self.format_prompt(inputs, input_critique))

        resp_critique = self.secondary.critique(response)
        resp_correction = self.secondary.correct(response, resp_critique)

        return resp_correction

