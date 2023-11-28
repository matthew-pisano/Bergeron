import os
from typing import Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import TextInput, EncodedInput, PreTokenizedInput

from src.models.model_utils import SpoofTokenizer
from src.logger import root_logger


class DevModel(PreTrainedModel):

    def __init__(self, model_name: str, **kwargs):

        if not model_name.startswith("dev"):
            raise ValueError("Dev models must have names in the form of 'dev/*'")

        self.name_or_path = model_name
        self.tokenizer = DevTokenizer(model_name)

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        """Spoofs the pretrained model generation to make it fit for API generation"""

        responses = []

        for encoded_prompt in inputs:
            prompt = self.tokenizer.decode(encoded_prompt.tolist())

            if self.name_or_path.endswith("human"):
                resp = self.manual_input(prompt)
            elif self.name_or_path.endswith("echo"):
                resp = self.generate_echo(prompt)
            else:
                raise ValueError(f"Could not find dev model with name '{self.name_or_path}'")

            responses.append(self.tokenizer.encode(resp))

        return torch.LongTensor(responses)

    @staticmethod
    def manual_input(prompt: str):
        root_logger.unchecked("[MANUAL PROMPT]\n", prompt)
        root_logger.info("[MANUAL INSTRUCTIONS] Enter ':s' to submit your response")

        resp = ""
        while True:
            partial_resp = input("> ").replace("> ", "")
            if partial_resp == ":s":
                break
            elif partial_resp.startswith(":") and len(partial_resp) == 2:
                raise ValueError(f"Unrecognized command '{partial_resp}'")

            resp += partial_resp+"\n"

        return resp.rstrip("\n")

    @staticmethod
    def generate_echo(prompt: str):
        return prompt


class DevTokenizer(PreTrainedTokenizer):

    def __init__(self, tokenizer_name: str, **kwargs):
        self.tokenizer_name = tokenizer_name

    def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], **kwargs):
        """Spoofs the pretrained tokenizer's encoding by converting characters to integers"""

        return SpoofTokenizer.char_encode(text, **kwargs)

    def decode(self, token_ids: Union[int, list[int]], **kwargs):
        """Spoofs the pretrained tokenizer's decoding by converting integers to characters"""

        return SpoofTokenizer.char_decode(token_ids, **kwargs)
