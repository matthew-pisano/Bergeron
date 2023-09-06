from typing import Optional, Union

import torch
import openai
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import TextInput, EncodedInput, PreTokenizedInput


class OpenAIModel(PreTrainedModel):

    def __init__(self, model_name: str, **kwargs):
        self.name_or_path = model_name
        self.tokenizer = OpenAITokenizer(model_name)

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        responses = []

        for encoded_prompt in inputs:
            prompt = self.tokenizer.decode(encoded_prompt.tolist())
            resp = openai.Completion.create(model=self.name_or_path, prompt=prompt)
            responses.append(self.tokenizer.encode(resp["choices"]["text"]))

        return torch.LongTensor(responses)


class OpenAITokenizer(PreTrainedTokenizer):

    def __init__(self, tokenizer_name: str, **kwargs):
        self.tokenizer_name = tokenizer_name

    def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], **kwargs):
        if type(text) is str:
            text = [text]

        encoded = []
        for seq in text:
            for char in seq:
                encoded.append(ord(char))

        return encoded

    def decode(self, token_ids: Union[int, list[int]], **kwargs):
        if type(token_ids) is int:
            token_ids = [token_ids]

        decoded = ""
        for char_id in token_ids:
            decoded += chr(int(char_id))

        return decoded
