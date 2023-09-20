import datetime
import os
import random
import time

import openai
from dotenv import load_dotenv
from transformers import LlamaForCausalLM, LlamaTokenizer, set_seed

from src.logger import root_logger
from src.models.base_model import BaseModel
from src.models.combined import Combined
from src.models.model_utils import ModelSrc, ModelInfo
from src.models.primary import Primary
from src.models.secondary import Secondary

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def converse(model: BaseModel):

    print("Enter ':q' to quit loop\nEnter ':s' to submit your response\nEnter ':r' to repeat the last non-command response")

    context = ""
    prev_context = ""
    while True:
        while True:
            response = input("> ").replace("> ", "")
            if response == ":q":
                return
            elif response == ":s":
                break
            elif response == ":r":
                context = prev_context + "\n"
                break
            elif response.startswith(":") and len(response) == 2:
                raise ValueError(f"Unrecognized command '{response}'")

            context += response + "\n"

        try:
            model_response = model.generate(context[:-1])
        except KeyboardInterrupt as e:
            print("Keyboard interrupt: canceling generation")
            continue

        print(model_response)
        prev_context = context
        context = ""


def main():
    main_start = time.time()
    print(f"Begin main at {datetime.datetime.utcfromtimestamp(main_start)} UTC")

    # root_logger.set_level(root_logger.DEBUG)
    rand = random.randint(0, 100)
    print(f"Setting random seed to {rand}")
    set_seed(rand)

    # model_name, model_src, model_class, tokenizer_class = "gpt-3.5-turbo", ModelSrc.OPENAI_API, None, None
    # model_name, model_src, model_class, tokenizer_class = "PY007/TinyLlama-1.1B-step-50K-105b", ModelSrc.LOCAL, LlamaForCausalLM, LlamaTokenizer
    # model_name, model_src, model_class, tokenizer_class = "Open-Orca/OpenOrca-Platypus2-13B", ModelSrc.HF_API, LlamaForCausalLM, LlamaTokenizer
    # model_name, model_src, model_class, tokenizer_class = "gpt2", ModelSrc.HF_API, None, None
    # model_name, model_src, model_class, tokenizer_class = "lmsys/vicuna-7b-v1.5", ModelSrc.LOCAL, LlamaForCausalLM, LlamaTokenizer
    # model_name, model_src, model_class, tokenizer_class = "meta-llama/Llama-2-7b-chat-hf", ModelSrc.LOCAL, LlamaForCausalLM, LlamaTokenizer
    model_name, model_src, model_class, tokenizer_class = "dev/human", ModelSrc.DEV, None, None

    model_info = ModelInfo(model_name, model_src, model_class, tokenizer_class)

    model_name, model_src, model_class, tokenizer_class = "dev/echo", ModelSrc.DEV, None, None
    rephrase_model_info = ModelInfo(model_name, model_src, model_class, tokenizer_class)

    primary = Primary(model_info)
    secondary = Secondary(model_info, rephrase_model_info)

    combined = Combined(primary, secondary)

    response = combined.generate("Write a poem about flowers")

    print(response)

    main_end = time.time()
    print(f"End main at {datetime.datetime.utcfromtimestamp(main_end)} UTC")
    print(f"Elapsed time of {round(main_end-main_start, 3)}s")


if __name__ == "__main__":
    main()
