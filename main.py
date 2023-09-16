import datetime
import os
import random
import time

import openai
from dotenv import load_dotenv
from transformers import LlamaForCausalLM, LlamaTokenizer

from src.models.base_model import BaseModel
from src.models.model_utils import ModelSrc, ModelInfo
from src.models.primary import Primary

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

    # model_name, model_src = "gpt-3.5-turbo", ModelSrc.OPENAI_API
    # model_name, model_src = "PY007/TinyLlama-1.1B-step-50K-105b", ModelSrc.LOCAL
    # model_name, model_src = "Open-Orca/OpenOrca-Platypus2-13B", ModelSrc.HF_API
    # model_name, model_src = "gpt2", ModelSrc.HF_API
    model_name, model_src = "lmsys/vicuna-7b-v1.5", ModelSrc.LOCAL

    model_info = ModelInfo(model_name, model_src, LlamaForCausalLM, LlamaTokenizer)
    primary = Primary(model_info)

    converse(primary)

    main_end = time.time()
    print(f"End main at {datetime.datetime.utcfromtimestamp(main_end)} UTC")
    print(f"Elapsed time of {round(main_end-main_start, 3)}s")


if __name__ == "__main__":
    main()
