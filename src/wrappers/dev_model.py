from typing import Optional

import torch
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateOutput

from src.wrappers.mock_tokenizer import MockTokenizer
from src.logger import root_logger


class DevModel(PreTrainedModel):
    """Developer Model wrapper.  Spoofs pretrained model generation while really generating text through manual input or predetermined methods"""

    def __init__(self, model_name: str, **kwargs):
        """
        Args:
            model_name: The name of the developer model to use"""

        if not model_name.startswith("dev"):
            raise ValueError("Dev models must have names in the form of 'dev/*'")

        self.name_or_path = model_name

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, **kwargs) -> GenerateOutput | torch.LongTensor:
        """Spoofs the pretrained model generation to make it fit for custom development generation

        Args:
            inputs: The input tokens to use for generation
        Returns:
            The generated response tokens"""

        tokenizer = MockTokenizer(self.name_or_path)
        responses = []

        for encoded_prompt in inputs:
            prompt = tokenizer.decode(encoded_prompt.tolist())

            if self.name_or_path.endswith("human"):
                resp = self.generate_manual(prompt)
            elif self.name_or_path.endswith("echo"):
                resp = self.generate_echo(prompt)
            else:
                raise ValueError(f"Could not find dev model with name '{self.name_or_path}'")

            responses.append(tokenizer.encode(resp))

        return torch.LongTensor(responses)

    @staticmethod
    def generate_manual(prompt: str):
        """Allows for users to generate responses to the prompt themselves through standard input for debugging purposes

        Args:
            prompt: The prompt to show to standard output
        Returns:
            The manually generated response"""

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
        """Simply echoes the prompt

        Args:
            prompt: The prompt to clone as the response

        Returns:
            The unchanged prompt itself as the response"""

        return prompt
