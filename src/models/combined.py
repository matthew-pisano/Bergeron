import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel
from src.models.primary import Primary
from src.models.secondary import Secondary
from src.logger import root_logger


class Combined(BaseModel):
    """The combined model architecture.  The primary model responds to user input as usual.  The secondary model vets both the input and the response"""

    def __init__(self, primary_model: Primary, secondary_model: Secondary):
        self.primary = primary_model
        self.secondary = secondary_model

    def generate(self, prompt: str, **kwargs):
        """Performs sanitizes the user input and evaluation of model output before returning the final response"""

        root_logger.info("Generating original response...")
        raw_response = self.primary.generate(prompt)

        root_logger.info("Rephrasing input...")
        rephrased_inputs = self.secondary.rephrase(prompt)
        root_logger.info("Critiquing inputs...")
        input_critique = self.secondary.critique(rephrased_inputs)

        if len(input_critique) > 0:
            root_logger.info("Generating conscience...")
            sanitized = self.insert_conscience(prompt, input_critique)
            root_logger.info("Generating response with conscience...")
            conscience_response = self.primary.generate(sanitized)

            root_logger.info("Generating comparison...")
            compare_response, sim_indicator, diff_indicator = self.secondary.compare(conscience_response, raw_response)

            if sim_indicator in compare_response:
                root_logger.info("Using original response")
            else:
                raw_response = conscience_response
                root_logger.info("Using conscience response")

        final_resp = raw_response
        root_logger.info("Generating final response critique...")
        resp_critique = self.secondary.critique(raw_response)

        if len(resp_critique) > 0:
            root_logger.info("Generating final correction...")
            final_resp = self.secondary.correct(raw_response, resp_critique)

        return final_resp

    @staticmethod
    def insert_conscience(prompt: str, prompt_critique: str):
        return f"""{prompt}

When looking at the above input, your conscience says {prompt_critique}
Take this into account as you respond to the prompt."""

