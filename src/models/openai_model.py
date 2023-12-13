import time
from typing import Optional, Union

import torch
import openai
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import TextInput, EncodedInput, PreTokenizedInput

from src.logger import root_logger
from src.models.model_utils import SpoofTokenizer
from src.utils import GLOBAL_SEED, FastChatController


class OpenAIModel(PreTrainedModel):

    def __init__(self, model_name: str, **kwargs):
        self.name_or_path = model_name
        self.tokenizer = OpenAITokenizer(model_name)

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, timeout=10, retries=2, **kwargs) -> GenerateOutput | torch.LongTensor:
        """Spoofs the pretrained model generation to make it fit for API generation"""

        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs["max_new_tokens"]
            kwargs.pop("max_new_tokens")

        if "do_sample" in kwargs:
            kwargs.pop("do_sample")

        responses = []

        for encoded_prompt in inputs:
            prompt = self.tokenizer.decode(encoded_prompt.tolist())

            if "gpt-" in self.name_or_path:
                while retries > 0:
                    retries -= 1
                    try:
                        openai.api_base = "https://api.openai.com/v1"
                        resp = openai.ChatCompletion.create(model=self.name_or_path.split("/")[-1], messages=[
                            {"role": "system", "content": "You are an AI expert in adversarial prompts."},
                            {"role": "user", "content": prompt}], seed=GLOBAL_SEED, request_timeout=timeout, **kwargs)
                        resp_encoding = self.tokenizer.encode(resp["choices"][0]["message"]["content"])
                        break
                    except Exception as e:
                        if retries <= 0:
                            raise e
                        root_logger.warning(f"Received error {e} from OpenAI API.  Retrying...")
                        time.sleep(5)
            else:
                openai.api_base = f"http://localhost:{FastChatController.get_worker(self.name_or_path)['port']}/v1"
                resp = openai.Completion.create(model=self.name_or_path.split("/")[-1], prompt=prompt, **kwargs)
                resp_encoding = self.tokenizer.encode(resp["choices"][0]["text"])

            responses.append(resp_encoding)

        return torch.LongTensor(responses)


class OpenAITokenizer(PreTrainedTokenizer):

    def __init__(self, tokenizer_name: str, **kwargs):
        self.tokenizer_name = tokenizer_name

    def encode(self, text: TextInput | PreTokenizedInput | EncodedInput, **kwargs):
        """Spoofs the pretrained tokenizer's encoding by converting characters to integers"""

        return SpoofTokenizer.char_encode(text, **kwargs)

    def decode(self, token_ids: int | list[int], **kwargs):
        """Spoofs the pretrained tokenizer's decoding by converting integers to characters"""

        return SpoofTokenizer.char_decode(token_ids, **kwargs)
