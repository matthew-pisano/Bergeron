import torch
import transformers.utils
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel


class Secondary(BaseModel):
    """The conscience model.  This examines both the user input and model output for alignment violations"""

    def __init__(self, critique_model_name: str, summarizer_model_name: str):
        self.critique_model, self.critique_tokenizer = self.from_pretrained(critique_model_name)

        self.summarizer_model, self.summarizer_tokenizer = self.from_pretrained(summarizer_model_name)

    def summarize(self, full_text: str, **kwargs):
        """Summarizes the given text"""

        return self.generate_using(full_text, self.summarizer_model, self.summarizer_tokenizer, **kwargs)

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
