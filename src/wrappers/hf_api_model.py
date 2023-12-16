import json
import os
import time
from typing import Optional

import torch
from huggingface_hub import InferenceClient
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateOutput

from src.wrappers.mock_tokenizer import MockTokenizer
from src.logger import root_logger


class HFAPIModel(PreTrainedModel):
    """Huggingface API Model wrapper.  Spoofs pretrained model generation while really generating text through the Huggingface API"""

    def __init__(self, model_name: str, model_task: str, **kwargs):
        """
        Args:
            model_name: The name of the huggingface model to use
            model_task: The name of the huggingface model task to perform"""

        self.name_or_path = model_name
        self.model_task = model_task

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, do_sample=True, temperature=0.7, max_new_tokens=None, timeout=30, retries=2, **kwargs) -> GenerateOutput | torch.LongTensor:
        """Spoofs the pretrained model generation to make it fit for huggingface API generation

        Args:
            inputs: The input tokens to use for generation
            do_sample: Whether to use the sampling decoding method
            temperature: The temperature of the model
            max_new_tokens: The maximum number of new tokens to generate
            timeout: The timeout for API requests
            retries: The number of retries to perform after an API error before throwing an exception
        Returns:
            The generated response tokens"""

        inference_client = InferenceClient(model=self.name_or_path, token=os.environ.get("HF_API_KEY"), timeout=timeout)

        tokenizer = MockTokenizer(self.name_or_path)
        responses = []

        # Set default params depending on the huggingface task to perform
        hf_params = {}
        if self.model_task == "conversational":
            hf_params = {"max_new_tokens": max_new_tokens, "return_full_text": False, "repetition_penalty": 1.5, "do_sample": do_sample}
        elif self.model_task == "summarization":
            hf_params = {"max_new_tokens": 250}

        for encoded_prompt in inputs:
            prompt = tokenizer.decode(encoded_prompt.tolist())

            # Truncates prompts to meet the stricter requirements of the API
            char_limit = 450
            if len(prompt) > char_limit:
                root_logger.warning(f"Prompt given to Huggingface API is too long! {len(prompt)} > {char_limit}.  This prompt will be truncated.")
                prompt = prompt[:char_limit//2] + prompt[-char_limit//2:]

            # Loop until a response is successfully generated from the API or the number of retries runs out
            resp_encoding = None
            while retries > 0:
                retries -= 1
                try:
                    resp = json.loads(inference_client.post(json={"inputs": prompt, "parameters": hf_params}).decode())
                    resp_encoding = tokenizer.encode(resp[0]["generated_text"])
                    if 'error' not in resp:
                        break
                    if retries <= 0:
                        raise RuntimeError(f"Received error from HuggingFace API: {resp['error']}")
                except Exception as e:
                    if retries <= 0:
                        raise e
                    root_logger.warning(f"Received error {e} from HuggingFace API.  Retrying...")
                    time.sleep(10)

            if resp_encoding is None:
                raise ValueError("Response encoding has not been properly generated")
            responses.append(resp_encoding)

        return torch.LongTensor(responses)
