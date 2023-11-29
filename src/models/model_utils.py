from enum import Enum
from typing import Union

from transformers import PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput


class ModelSrc(Enum):

    LOCAL = "local"
    OPENAI_API = "openai"
    HF_API = "huggingface_hub"
    DEV = "dev"


class ModelInfo:

    def __init__(self, pretrained_model_name_or_path: str, model_src: ModelSrc, model_class: PreTrainedModel | None = AutoModelForCausalLM, tokenizer_class: PreTrainedTokenizer | None = AutoTokenizer, model_task: str = None):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_src = model_src
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.model_task = model_task

        if self.model_src == ModelSrc.HF_API and self.model_task is None:
            raise ValueError("A model task is required to use HuggingFace models")

    def as_dict(self):
        return vars(self)


class SpoofTokenizer:
    @staticmethod
    def char_encode(text: Union[TextInput, PreTokenizedInput, EncodedInput], **kwargs):
        """Spoofs the pretrained tokenizer's encoding by converting characters to integers"""

        if type(text) is str:
            text = [text]

        encoded = []
        for seq in text:
            for char in seq:
                encoded.append(ord(char))

        return encoded

    @staticmethod
    def char_decode(token_ids: Union[int, list[int]], **kwargs):
        """Spoofs the pretrained tokenizer's decoding by converting integers to characters"""

        if type(token_ids) is int:
            token_ids = [token_ids]

        decoded = ""
        for char_id in token_ids:
            decoded += chr(int(char_id))

        return decoded
