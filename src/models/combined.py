import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel
from src.models.primary import Primary
from src.models.secondary import Secondary


class Combined(BaseModel):
    """The combined model architecture.  The primary model responds to user input as usual.  The secondary model vets both the input and the response"""

    def __init__(self, primary_model: Primary, secondary_model: Secondary):
        self.primary = primary_model
        self.secondary = secondary_model

    def generate(self, prompt: str, **kwargs):
        """Performs sanitizes the user input and evaluation of model output before returning the final response"""

        raw_response = self.primary.generate(prompt)

        rephrased_inputs = self.secondary.rephrase(prompt)
        input_critique = self.secondary.critique(rephrased_inputs)

        if len(input_critique) > 0:
            sanitized = self.insert_conscience(prompt, input_critique)
            conscience_response = self.primary.generate(sanitized)

            compare_response, sim_indicator, diff_indicator = self.secondary.compare(conscience_response, raw_response)

            raw_response = raw_response if sim_indicator in compare_response else conscience_response

        final_resp = raw_response
        resp_critique = self.secondary.critique(raw_response)

        if len(resp_critique) > 0:
            final_resp = self.secondary.correct(raw_response, resp_critique)

        return final_resp

    @staticmethod
    def insert_conscience(prompt: str, prompt_critique: str):
        return f"""{prompt}

When looking at the above input, your conscience says {prompt_critique}
Take this into account as you respond to the prompt."""

