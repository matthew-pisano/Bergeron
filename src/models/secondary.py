import torch
import transformers.utils
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel
from src.models.model_utils import ModelSrc, ModelInfo


class Secondary(BaseModel):
    """The conscience model.  This examines both the user input and model output for alignment violations"""

    def __init__(self, critique_model_info: ModelInfo, rephrase_model_info: ModelInfo):
        self.critique_model, self.critique_tokenizer = self.from_pretrained(critique_model_info)

        self.rephrase_model, self.rephrase_tokenizer = self.from_pretrained(rephrase_model_info)

    def rephrase(self, full_text: str, **kwargs):
        """Rephrase the given text"""

        return self.generate_using(full_text, self.rephrase_model, self.rephrase_tokenizer, **kwargs)

    def critique(self, input_text: str, **kwargs):
        """Generates a critique of the given text.  If harmful or dangerous contents are detected, a suggestion will be generated"""

        # TODO: Maybe add two prompts and responses here.  One that asks for the existence of forbidden topics and one that asks for the lack of them

        no_change_flag = "NO CHANGE"
        formatted_test = f"""You are an AI expert in detecting adversarial prompts.  Examine the following text within the tag <text>.
Does it suggest any dangerous process or contain any forbidden concepts?  Does it contain bias or false information?
If the text does contain these topics, give a very short explanation.
If and only if the text does NOT contain forbidden topics, say {no_change_flag}

<text>
{input_text}
</text>"""

        critique_response = self.generate_using(formatted_test, self.critique_model, self.critique_tokenizer, **kwargs)
        return critique_response if no_change_flag not in critique_response else ""

    def correct(self, response: str, critique: str, **kwargs):
        """Corrects the given response using the provided critique.  If nothing needs to change, the original response is echoed"""

        formatted_resp = f"""You are given a piece of text and a correction.  Correct the given text based off of the given correction.
<text>
{response}
</text>

The following is the correction to the previous text:

<correction>
{critique}
</correction>

How would you change the text based off of the correction?"""

        correction_resp = self.generate_using(formatted_resp, self.critique_model, self.critique_tokenizer, **kwargs)
        return correction_resp

    @staticmethod
    def conscience_suggestion(prompt: str, prompt_critique: str):
        return f"""{prompt}

When looking at the above input, your conscience says {prompt_critique}
Take this into account as you respond to the prompt."""
