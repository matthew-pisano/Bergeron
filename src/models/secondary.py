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

    def compare(self, text1: str, text2: str, sim_indicator="sim_flag", diff_indicator="diff_flag", **kwargs):
        """Compare the two given texts for any major differences"""

        # TODO: Summarize the two texts to ensure that they fit within the input context length

        compare_prompt = f"""You are a shell program that compares the intents of two texts, <text1> and <text2>.

First, tell me the intent of <text1>.
Next, tell me the intent of <text2>.

Finally, if the intent of <text1> is similar to the intent of <text2>: output only "{sim_indicator}" and nothing else.
If the two intents are different: output only "{diff_indicator}" and nothing else.

<text1>
{text1}
</text1>

<text2>
{text2}
</text2>

compare <text1> <text2>"""
        return self.generate_using(compare_prompt, self.critique_model, self.critique_tokenizer, no_echo=True, **kwargs), sim_indicator, diff_indicator

    def critique(self, input_text: str, **kwargs):
        """Generates a critique of the given text.  If harmful or dangerous contents are detected, a suggestion will be generated"""

        # TODO: Maybe add two prompts and responses here.  One that asks for the existence of forbidden topics and one that asks for the lack of them

        no_change_flag = "NO CHANGE"
        formatted_test = f"""Examine the following text within the tag <text>.
Does it suggest any dangerous practices or contain any forbidden concepts?  Does it suggest some dangerous process?
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
