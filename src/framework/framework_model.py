import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.constants import logger


class FrameworkModel:
    """Base class for all framework models"""

    @property
    def name(self):
        """The name of the underlying model or models"""
        raise NotImplementedError(f"Name is not implemented for class {self.__class__.__name__}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generates a response to the prompt

        Args:
            prompt: The prompt to generate a response for
        Returns:
            The generated response string"""

        raise NotImplementedError(f"Generate is not implemented for class {self.__class__.__name__}")

    @staticmethod
    def generate_using(prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, no_echo=True, do_sample=True, temperature=0.7, max_new_tokens=128, **kwargs):
        """Utility interface for generating a response from a transformers library model. Automatically handles encoding and decoding

        Args:
            prompt: The prompt to generate a response for
            model: The model to give the prompt to
            tokenizer: The tokenizer to use for encoding and decoding
            no_echo: Whether to strip out the prompt if it is at the beginning of the prompt
            do_sample: Whether to use the sampling decoding method
            temperature: The temperature of the model
            max_new_tokens: The maximum number of new tokens to generate
        Returns:
            The generated response string"""

        if max_new_tokens is None:
            max_new_tokens = 128

        prompt_tokens = tokenizer.encode(prompt)

        generated = model.generate(torch.Tensor([prompt_tokens]).int(), do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens, **kwargs)[0]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        if no_echo and not model.name_or_path.startswith("dev") and prompt in response:
            response = response.replace(prompt, "")

        logger.debug(f"[MODEL ({model.name_or_path}) PROMPT]\n%s", prompt)
        logger.debug(f"[MODEL ({model.name_or_path}) RESP]\n%s", response)

        return response.strip("\n")

    def __repr__(self):
        return self.name
