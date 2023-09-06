from typing import Optional, Union

import torch
import openai
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import TextInput, EncodedInput, PreTokenizedInput


class OpenAIModel(PreTrainedModel):

    def __init__(self, model_name: str, **kwargs):
        self.name_or_path = model_name

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        responses = []

        for prompt in inputs:
            resp = openai.Completion.create(model=self.name_or_path, prompt=prompt)
            responses.append(resp["choices"]["text"])

        return torch.LongTensor(responses)


class OpenAITokenizer(PreTrainedTokenizer):

    def __init__(self, tokenizer_name: str, **kwargs):
        self.tokenizer_name = tokenizer_name

    def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], **kwargs) -> list[int]:
        raise NotImplementedError()

    def decode(self, token_ids: Union[int, list[int]], **kwargs) -> list[int]:
        raise NotImplementedError()
