from enum import Enum
from typing import Union

from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, EncodedInput


class ModelSrc(Enum):

    LOCAL = "local"
    OPENAI_API = "openai"
    HF_API = "huggingface_hub"


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