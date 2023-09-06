import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.primary import Primary
from src.models.secondary import Secondary


class Combined:

    def __init__(self, primary_model: Primary, secondary_model: Secondary):
        self.primary = primary_model
        self.secondary = secondary_model

    def generate(self, inputs: list[str], **kwargs):

        summarized_inputs = self.secondary.summarize(inputs)

        input_critique = self.secondary.critique(summarized_inputs)

        self.insert_conscience(inputs, input_critique)

        response = self.primary.generate(inputs)

        resp_critique = self.secondary.critique(response)
        resp_correction = self.secondary.correct(response, resp_critique)

        return resp_correction

    @staticmethod
    def insert_conscience(inputs: torch.Tensor, input_critique: torch.Tensor):
        for i in range(len(inputs)):
            inputs[i] = \
f"""{inputs[i]}

When looking at the above input, your conscience says {input_critique[i]}
Take this into account as you respond to the prompt."""

