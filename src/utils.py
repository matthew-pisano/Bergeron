import os
import subprocess
import time
from subprocess import Popen

import openai
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.logger import root_logger
from src.framework.base_model import BaseModel
from src.models.model_utils import ModelSrc
from src.models.openai_model import OpenAIModel


_fastchat_workers = []


def use_fastchat_model(model_path: str):
    if model_path in _fastchat_workers:
        return

    if len(_fastchat_workers) == 0:
        root_logger.info("Initializing fastchat controller...")
        Popen(['python3', '-m', 'fastchat.serve.controller'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(3)

    root_logger.info(f"Initializing {model_path} worker...")
    Popen(['python3', '-m', 'fastchat.serve.model_worker', '--model-path', model_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(10)

    if len(_fastchat_workers) == 0:
        root_logger.info("Starting fastchat openai server...")
        Popen(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', 'localhost', '--port', '8000'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        time.sleep(15)

    root_logger.info("Started!")
    openai.api_base = "http://localhost:8000/v1"

    _fastchat_workers.append(model_path)


def set_seed(seed: int):
    print(f"Setting random seed to {seed}")
    transformers.set_seed(seed)
    OpenAIModel.set_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def converse(model: BaseModel, do_sample=True, temperature=0.7, max_new_tokens=None, **kwargs):

    print("Enter ':q' to quit loop\nEnter ':s' to submit your response\nEnter ':r' to repeat the last non-command response")

    context = ""
    prev_context = ""
    while True:
        while True:
            response = input("> ")
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
            model_response = model.generate(context[:-1], do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens, **kwargs)
        except KeyboardInterrupt as e:
            print("Keyboard interrupt: canceling generation")
            continue

        print(model_response)
        prev_context = context
        context = ""


def model_info_from_name(target_model_name: str) -> tuple[str, ModelSrc, PreTrainedModel | None, PreTrainedTokenizer | None]:
    if target_model_name.startswith("dev/"):
        model_name, model_src, model_class, tokenizer_class = target_model_name, ModelSrc.DEV, None, None
    elif target_model_name.startswith("meta-llama/") or target_model_name.startswith("mistralai/"):
        model_name, model_src, model_class, tokenizer_class = target_model_name, ModelSrc.OPENAI_API, None, None
        use_fastchat_model(model_name)
    elif "gpt-" in target_model_name:
        model_name, model_src, model_class, tokenizer_class = target_model_name, ModelSrc.OPENAI_API, None, None
    else:
        raise ValueError(f"Unknown model name '{target_model_name}'")

    return model_name, model_src, model_class, tokenizer_class
