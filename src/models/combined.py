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

        prompt = prompt.strip("\n").strip(" ")
        root_logger.info("Generating original response...")
        primary_response = self.primary.generate(prompt, **kwargs)

        root_logger.info("Rephrasing input...")
        rephrased_inputs = self.secondary.rephrase(prompt)

        root_logger.info("Critiquing inputs...")
        input_critique = self.secondary.critique(rephrased_inputs, **kwargs)

        if len(input_critique) > 0:
            root_logger.info("Generating conscience...")
            sanitized = self.insert_conscience(prompt, input_critique)

            root_logger.info("Generating response with conscience...")
            primary_response = self.primary.generate(sanitized, **kwargs)

        root_logger.info("Generating final response critique...")
        resp_critique = self.secondary.critique(primary_response, **kwargs)

        if len(resp_critique) > 0:
            root_logger.info("Generating final correction...")
            primary_response = self.secondary.correct(primary_response, resp_critique, **kwargs)

        return primary_response

    @staticmethod
    def insert_conscience(prompt: str, prompt_critique: str):
        return f"""{prompt}

When looking at the above input, your conscience says {prompt_critique}
Take this into account as you respond to the prompt."""

