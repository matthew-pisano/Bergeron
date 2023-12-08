import dataclasses

from src.framework.base_model import BaseModel
from src.framework.primary import Primary
from src.framework.secondary import Secondary
from src.logger import root_logger


@dataclasses.dataclass
class DetectionReport:
    """Class mutated based on detections of any dangerous text during a combined model's generation"""

    dangerous_prompt: bool = False
    dangerous_response: bool = False
    sealed: bool = False

    def seal(self):
        self.sealed = True

    def __setattr__(self, key, value):
        if key == "sealed" and value is False and self.sealed is True:
            raise AttributeError("Cannot unseal a sealed report")
        if self.sealed:
            raise AttributeError("Cannot set the attribute of a report once it has been sealed")
        super().__setattr__(key, value)


class Combined(BaseModel):
    """The combined model architecture.  The primary model responds to user input as usual.  The secondary model vets both the input and the response"""

    def __init__(self, primary_model: Primary, secondary_model: Secondary):
        self.primary = primary_model
        self.secondary = secondary_model

    @property
    def name(self):
        return f"C({self.primary.model.name_or_path}, {self.secondary.critique_model.name_or_path})"

    def generate(self, prompt: str, detection_report: DetectionReport = None, **kwargs):
        """Performs sanitizes the user input and evaluation of model output before returning the final response"""

        prompt = prompt.strip("\n").strip(" ")

        root_logger.debug("Rephrasing prompt...")
        rephrased_prompt = self.secondary.rephrase(prompt)

        root_logger.debug("Critiquing prompt...")
        input_critique = self.secondary.critique_prompt(rephrased_prompt, **kwargs)

        if len(input_critique) > 0:
            if detection_report is not None:
                detection_report.dangerous_prompt = True

            root_logger.debug("Generating conscience...")
            sanitized = self.secondary.make_conscience_prompt(prompt, input_critique)

            root_logger.debug("Generating response with conscience...")
            primary_response = self.primary.generate(sanitized, **kwargs)
        else:
            root_logger.debug("Generating unfiltered input response...")
            primary_response = self.primary.generate(prompt, **kwargs)

        root_logger.debug("Generating final response critique...")
        resp_critique = self.secondary.critique_response(primary_response, **kwargs)

        if len(resp_critique) > 0:
            if detection_report is not None:
                detection_report.dangerous_response = True

            root_logger.debug("Generating final correction...")
            correction_prompt = self.secondary.make_correction_prompt(primary_response, resp_critique)
            primary_response = self.primary.generate(correction_prompt, **kwargs)

        detection_report.seal()

        return primary_response
