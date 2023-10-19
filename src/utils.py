import os
import random
import subprocess
import time
from subprocess import Popen

import numpy.random
import openai
import torch
import transformers

from src.logger import root_logger
from src.models.base_model import BaseModel


def use_fastchat_model(model_path: str):
    root_logger.info("Initializing fastchat controller...")
    Popen(['python3', '-m', 'fastchat.serve.controller'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(3)
    root_logger.info(f"Initializing {model_path} worker...")
    Popen(['python3', '-m', 'fastchat.serve.model_worker', '--model-path', model_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(10)
    root_logger.info("Starting fastchat openai server...")
    Popen(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', 'localhost', '--port', '8000'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    time.sleep(5)
    root_logger.info("Started!")
    openai.api_base = "http://localhost:8000/v1"


def set_seed(seed: int):
    print(f"Setting random seed to {seed}")
    # set_seed(rand)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
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

