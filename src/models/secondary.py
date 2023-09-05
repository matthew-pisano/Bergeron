import torch
import transformers.utils
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class Secondary:

    def __init__(self, pretrained_critique_name_or_path: str, pretrained_summarizer_name_or_path: str):
        self.critique_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained_critique_name_or_path)
        self.critique_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_critique_name_or_path)

        self.summarizer_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(pretrained_summarizer_name_or_path)
        self.summarizer_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(pretrained_summarizer_name_or_path)

    def summarize(self, inputs: torch.Tensor, **kwargs):
        return self.summarizer_model.generate(inputs, **kwargs)

    def critique(self, inputs: torch.Tensor, **kwargs):
        return self.critique_model.generate(inputs, **kwargs)

    def correct(self, inputs: torch.Tensor, critique: torch.Tensor, **kwargs):
        return self.critique_model.generate(inputs, **kwargs)
