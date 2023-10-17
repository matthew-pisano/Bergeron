import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer, AutoModel

from src.models.hf_model import HFModel, HFTokenizer
from src.models.dev_model import DevModel, DevTokenizer
from src.models.model_utils import ModelSrc, ModelInfo
from src.models.openai_model import OpenAIModel, OpenAITokenizer
from src.logger import root_logger


class BaseModel:

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError(f"Generate is not implemented for class {self.__class__.__name__}")

    @staticmethod
    def generate_using(prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, no_echo=True, do_sample=True, temperature=0.7, max_length=512, **kwargs):
        """Performs the appropriate encoding and decoding to generate a string response to a string prompt"""

        if max_length is None:
            max_length = 512

        root_logger.debug("[MODEL PROMPT]\n", prompt)

        prompt_tokens = tokenizer.encode(prompt)

        generated = model.generate(torch.Tensor([prompt_tokens]).int(), do_sample=do_sample, temperature=temperature, max_length=max_length, **kwargs)[0]
        response = tokenizer.decode(generated, skip_special_tokens=True)

        # root_logger.debug("[MODEL ORIGINAL RESP]\n", response)

        if no_echo and not model.name_or_path.startswith("dev") and prompt in response:
            response = response.replace(prompt, "")

        # root_logger.debug("[MODEL RESP]\n", response)

        return response.strip("\n")

    @staticmethod
    def from_pretrained(model_info: ModelInfo) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Gets the pretrained model and tokenizer from the given model name"""

        root_logger.debug(f"Loading a pretrained model {model_info.pretrained_model_name_or_path} from {model_info.model_src}")
        if model_info.model_src == ModelSrc.OPENAI_API:
            return OpenAIModel(model_info.pretrained_model_name_or_path), OpenAITokenizer(model_info.pretrained_model_name_or_path)
        elif model_info.model_src == ModelSrc.HF_API:
            return HFModel(model_info.pretrained_model_name_or_path), HFTokenizer(model_info.pretrained_model_name_or_path)
        elif model_info.model_src == ModelSrc.DEV:
            return DevModel(model_info.pretrained_model_name_or_path), DevTokenizer(model_info.pretrained_model_name_or_path)
        else:
            try:
                model = model_info.model_class.from_pretrained(model_info.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
            except ValueError as e:
                root_logger.warning(f"Could not load {model_info.pretrained_model_name_or_path} as a {model_info.model_class} model.  Using AutoModel instead.")
                model = AutoModel.from_pretrained(model_info.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)

            return model, model_info.tokenizer_class.from_pretrained(model_info.pretrained_model_name_or_path)
