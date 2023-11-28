import os
import time
from typing import Optional, Union

import torch
from huggingface_hub.inference_api import InferenceApi
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import TextInput, EncodedInput, PreTokenizedInput

from src.models.model_utils import SpoofTokenizer
from src.logger import root_logger


class HFModel(PreTrainedModel):

    def __init__(self, model_name: str, **kwargs):
        self.name_or_path = model_name
        self.api_model = InferenceApi(repo_id=model_name, token=os.environ.get("HF_API_KEY"))
        self.tokenizer = HFTokenizer(model_name)

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, do_sample=True, temperature=0.7, max_new_tokens=None, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        """Spoofs the pretrained model generation to make it fit for API generation"""

        responses = []

        for encoded_prompt in inputs:
            prompt = self.tokenizer.decode(encoded_prompt.tolist())
            char_limit = 300
            if len(prompt) > char_limit:
                root_logger.warning(f"Prompt given to Huggingface API is too long! {len(prompt)} > {char_limit}.  This prompt will be truncated.")
                prompt = prompt[:char_limit//2] + prompt[-char_limit//2:]
            errors = 0
            while errors < 5:
                resp = self.api_model(inputs=prompt, params={"max_new_tokens": max_new_tokens, "return_full_text": False, "repetition_penalty": 1.5})
                if 'error' in resp:
                    errors += 1
                    root_logger.warning("Error from huggingface API!")
                    time.sleep(20)
                else:
                    break
            responses.append(self.tokenizer.encode(resp[0]["generated_text"]))

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
