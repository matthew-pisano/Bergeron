import json
import os
import time
from typing import Optional, Union

import torch
from huggingface_hub import InferenceClient
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import TextInput, EncodedInput, PreTokenizedInput

from src.models.model_utils import SpoofTokenizer
from src.logger import root_logger


class HFModel(PreTrainedModel):

    def __init__(self, model_name: str, model_task: str, **kwargs):
        self.name_or_path = model_name
        self.model_task = model_task
        self.tokenizer = HFTokenizer(model_name)

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, do_sample=True, temperature=0.7, max_new_tokens=None, timeout=30, retries=2, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        """Spoofs the pretrained model generation to make it fit for API generation"""

        inference_client = InferenceClient(model=self.name_or_path, token=os.environ.get("HF_API_KEY"), timeout=timeout)

        responses = []

        hf_params = {}
        if self.model_task == "conversational":
            hf_params = {"max_new_tokens": max_new_tokens, "return_full_text": False, "repetition_penalty": 1.5, "do_sample": do_sample}
        elif self.model_task == "summarization":
            hf_params = {"max_new_tokens": 250}

        for encoded_prompt in inputs:
            prompt = self.tokenizer.decode(encoded_prompt.tolist())
            char_limit = 300
            if len(prompt) > char_limit:
                root_logger.warning(f"Prompt given to Huggingface API is too long! {len(prompt)} > {char_limit}.  This prompt will be truncated.")
                prompt = prompt[:char_limit//2] + prompt[-char_limit//2:]

            while retries > 0:
                retries -= 1
                try:
                    resp = json.loads(inference_client.post(json={"inputs": prompt, "parameters": hf_params}).decode())
                    resp_encoding = self.tokenizer.encode(resp[0]["generated_text"])
                    if 'error' not in resp:
                        break
                    if retries <= 0:
                        raise RuntimeError(f"Received error from HuggingFace API: {resp['error']}")
                except Exception as e:
                    if retries <= 0:
                        raise e
                    root_logger.warning(f"Received error {e} from HuggingFace API.  Retrying...")
                    time.sleep(10)

            responses.append(resp_encoding)

        return torch.LongTensor(responses)


class HFTokenizer(PreTrainedTokenizer):

    def __init__(self, tokenizer_name: str, **kwargs):
        self.tokenizer_name = tokenizer_name

    def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], **kwargs):
        """Spoofs the pretrained tokenizer's encoding by converting characters to integers"""

        return SpoofTokenizer.char_encode(text, **kwargs)

    def decode(self, token_ids: Union[int, list[int]], **kwargs):
        """Spoofs the pretrained tokenizer's decoding by converting integers to characters"""

        return SpoofTokenizer.char_decode(token_ids, **kwargs)
