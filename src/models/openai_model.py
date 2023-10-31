import time
from typing import Optional, Union

import torch
import openai
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.utils import GenerateOutput
from transformers.tokenization_utils_base import TextInput, EncodedInput, PreTokenizedInput

from src.logger import root_logger
from src.models.model_utils import SpoofTokenizer


class OpenAIModel(PreTrainedModel):

    def __init__(self, model_name: str, **kwargs):
        self.name_or_path = model_name
        self.tokenizer = OpenAITokenizer(model_name)

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor] = None, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
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
                errors = 0
                while errors < 3:
                    try:
                        resp = openai.ChatCompletion.create(model=self.name_or_path.split("/")[-1], messages=[
                            {"role": "system", "content": "You are an AI expert in adversarial prompts."},
                            {"role": "user", "content": prompt}], **kwargs)
                        break
                    except Exception as e:
                        root_logger.error(f"Error calling OpenAIAPI: '{e}'")
                        errors += 1
                        time.sleep(30)

                resp_text = resp["choices"][0]["message"]["content"]
            else:
                resp = openai.Completion.create(model=self.name_or_path.split("/")[-1], prompt=prompt, **kwargs)
                resp_text = resp["choices"][0]["text"]

            # root_logger.debug("OpenAI resp", resp)
            responses.append(self.tokenizer.encode(resp_text))

        return torch.LongTensor(responses)


class OpenAITokenizer(PreTrainedTokenizer):

    def __init__(self, tokenizer_name: str, **kwargs):
        self.tokenizer_name = tokenizer_name

    def encode(self, text: Union[TextInput, PreTokenizedInput, EncodedInput], **kwargs):
        """Spoofs the pretrained tokenizer's encoding by converting characters to integers"""

        return SpoofTokenizer.char_encode(text, **kwargs)

    def decode(self, token_ids: Union[int, list[int]], **kwargs):
        """Spoofs the pretrained tokenizer's decoding by converting integers to characters"""

        return SpoofTokenizer.char_decode(token_ids, **kwargs)
