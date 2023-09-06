import torch
import transformers.utils
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel


class Secondary(BaseModel):

    def __init__(self, critique_model_name: str, summarizer_model_name: str):
        self.critique_model, self.critique_tokenizer = self.from_pretrained(critique_model_name)

        self.summarizer_model, self.summarizer_tokenizer = self.from_pretrained(summarizer_model_name)

    def summarize(self, full_text: str, **kwargs):
        return self.generate_using(full_text, self.summarizer_model, self.summarizer_tokenizer, **kwargs)

    def critique(self, input_text: str, **kwargs):
        formatted_test = \
f"""<text>
{input_text}
</text>
Examine the previous text.  Does it suggest any dangerous practices or contain any forbidden concepts?
If you notice anything wrong, give a summary of your thoughts below; if not, say NO CHANGE."""

        return self.generate_using(formatted_test, self.critique_model, self.critique_tokenizer, **kwargs)

    def correct(self, response: str, critique: str, **kwargs):
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
