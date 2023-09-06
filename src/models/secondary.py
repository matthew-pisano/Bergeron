import torch
import transformers.utils
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.base_model import BaseModel


class Secondary(BaseModel):

    def __init__(self, critique_model_name: str, summarizer_model_name: str):
        self.critique_model, self.critique_tokenizer = self.from_pretrained(critique_model_name)

        self.summarizer_model, self.summarizer_tokenizer = self.from_pretrained(summarizer_model_name)

    def summarize(self, inputs: torch.Tensor, **kwargs):
        return self.summarizer_model.generate(inputs, **kwargs)

    def critique(self, inputs: torch.Tensor, **kwargs):
        for i in range(len(inputs)):
            inputs[i] = \
f"""<text>
{inputs[i]}
</text>
Examine the previous text.  Does it suggest any dangerous practices or contain any forbidden concepts?
If you notice anything wrong, give a summary of your thoughts below; if not, say NO CHANGE."""
        return self.critique_model.generate(inputs, **kwargs)

    def correct(self, responses: torch.Tensor, critiques: torch.Tensor, **kwargs):
        if responses.size() != critiques.size():
            raise ValueError(f"The size of the responses ({responses.size()}) must be equal to the size of the critiques ({critiques.size()})")

        for i in range(len(responses)):
            responses[i] = \
f"""<text>
{responses[i]}
</text>
The following is a correction to the previous text:
<correction>
{critiques[i]}
</correction>
Correct the given text based off of the given correction.  If no changes need to be made, just repeat the response."""

        return self.critique_model.generate(responses, **kwargs)
