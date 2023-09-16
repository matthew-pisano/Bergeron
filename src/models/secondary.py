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

    def compare(self, text1: str, text2: str, **kwargs):
        """Compare the two given texts for any major differences"""

        compare_prompt = f"""You are a shell program that compares two texts.  If text1 is similar to text2: output only 1, if they are not: output only 0.
<text1>
{text1}
</text1>
<text2>
{text2}
</text2>"""
        return self.generate_using(compare_prompt, self.critique_model, self.critique_tokenizer, **kwargs)

    def critique(self, input_text: str, **kwargs):
        """Generates a critique of the given text.  If harmful or dangerous contents are detected, a suggestion will be generated"""

        formatted_test = \
f"""<text>
{input_text}
</text>
Examine the previous text.  Does it suggest any dangerous practices or contain any forbidden concepts?
If you notice anything wrong, give a summary of your thoughts below; if not, say NO CHANGE."""

        return self.generate_using(formatted_test, self.critique_model, self.critique_tokenizer, **kwargs)

    def correct(self, response: str, critique: str, **kwargs):
        """Corrects the given response using the provided critique.  If nothing needs to change, the original response is echoed"""

        formatted_resp = \
f"""<text>
{response}
</text>
The following is a correction to the previous text:
<correction>
{critique}
</correction>
Correct the given text based off of the given correction.  If no changes need to be made, just repeat the response."""

        return self.generate_using(formatted_resp, self.critique_model, self.critique_tokenizer, **kwargs)
