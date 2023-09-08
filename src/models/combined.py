import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.primary import Primary
from src.models.secondary import Secondary


class Combined:
    """The combined model architecture.  The primary model responds to user input as usual.  The secondary model vets both the input and the response"""

    def __init__(self, primary_model: Primary, secondary_model: Secondary):
        self.primary = primary_model
        self.secondary = secondary_model

    def generate(self, prompt: str, **kwargs):
        """Performs sanitizes the user input and evaluation of model output before returning the final response"""

        rephrased_inputs = self.secondary.rephrase(prompt)

        input_critique = self.secondary.critique(rephrased_inputs)

        sanitized = self.insert_conscience(prompt, input_critique)

        conscience_response = self.primary.generate(sanitized)
        raw_response = self.primary.generate(prompt)

        compare_result = self.secondary.compare(conscience_response, raw_response)

        # TODO: Branch program at comparison evaluation.  If they are the same, evaluate the raw response.  If they are different, evaluate the conscience response
        to_critique = ...

        resp_critique = self.secondary.critique(to_critique)
        resp_correction = self.secondary.correct(to_critique, resp_critique)

        return resp_correction

    @staticmethod
    def insert_conscience(prompt: str, prompt_critique: str):
        return f"""{prompt}

When looking at the above input, your conscience says {prompt_critique}
Take this into account as you respond to the prompt."""

