import time
from typing import Optional

import torch
import openai
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateOutput

from src.logger import root_logger
from src.wrappers.mock_tokenizer import MockTokenizer
from src.utils import GLOBAL_SEED
from src.fastchat import FastChatController


class OpenAIAPIModel(PreTrainedModel):
    """OpenAI API Model wrapper.  Spoofs pretrained model generation while really generating text through the OPENAI API"""

    def __init__(self, model_name: str, **kwargs):
        """
        Args:
            model_name: The name of the OpenAI model to use"""

        self.name_or_path = model_name

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, timeout=10, retries=2, **kwargs) -> GenerateOutput | torch.LongTensor:
        """Spoofs the pretrained model generation to make it fit for OpenAI API generation

        Args:
            inputs: The input tokens to use for generation
            timeout: The timeout for API requests
            retries: The number of retries to perform after an API error before throwing an exception
        Returns:
            The generated response tokens"""

        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs["max_new_tokens"]
            kwargs.pop("max_new_tokens")

        if "do_sample" in kwargs:
            kwargs.pop("do_sample")

        tokenizer = MockTokenizer(self.name_or_path)
        responses = []

        for encoded_prompt in inputs:
            prompt = tokenizer.decode(encoded_prompt.tolist())

            resp_encoding = None

            # Generation from the normal OpenAI API
            if "gpt-" in self.name_or_path:

                # Loop until a response is successfully generated from the API or the number of retries runs out
                while retries > 0:
                    retries -= 1
                    try:
                        openai.api_base = "https://api.openai.com/v1"
                        resp = openai.ChatCompletion.create(model=self.name_or_path.split("/")[-1], messages=[
                            {"role": "system", "content": "You are an AI expert in adversarial prompts."},
                            {"role": "user", "content": prompt}], seed=GLOBAL_SEED, request_timeout=timeout, **kwargs)
                        resp_encoding = tokenizer.encode(resp["choices"][0]["message"]["content"])
                        break
                    except Exception as e:
                        if retries <= 0:
                            raise e
                        root_logger.warning(f"Received error {e} from OpenAI API.  Retrying...")
                        time.sleep(5)
            # Generation from the fastchat API
            else:
                openai.api_base = f"http://localhost:{FastChatController.get_worker(self.name_or_path)['port']}/v1"
                resp = openai.Completion.create(model=self.name_or_path.split("/")[-1], prompt=prompt, **kwargs)
                resp_encoding = tokenizer.encode(resp["choices"][0]["text"])

            if resp_encoding is None:
                raise ValueError("Response encoding has not been properly generated")
            responses.append(resp_encoding)

        return torch.LongTensor(responses)
