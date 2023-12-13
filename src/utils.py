import os
import subprocess
import time
from subprocess import Popen

import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.logger import root_logger
from src.models.model_utils import ModelSrc


GLOBAL_SEED = None


class FastChatController:

    _workers = {}
    controller_process = None

    port_generator = (i for i in range(8000, 8005))

    @classmethod
    def get_worker(cls, model_path: str):
        return cls._workers[model_path]

    @classmethod
    def open(cls, model_path: str, port: int = None):

        if model_path in cls._workers:
            return

        if port is None:
            port = next(cls.port_generator)

        if cls.controller_process is None:
            root_logger.info("Initializing fastchat controller...")
            cls.controller_process = Popen(['python3', '-m', 'fastchat.serve.controller'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            time.sleep(3)

        if model_path not in cls._workers:
            root_logger.info(f"Initializing {model_path} worker...")
            worker_process = Popen(['python3', '-m', 'fastchat.serve.model_worker', '--model-path', model_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            time.sleep(10)

            root_logger.info("Starting fastchat openai server...")
            server_process = Popen(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', 'localhost', '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            time.sleep(15)

            cls._workers[model_path] = {"port": port, "worker_process": worker_process, "server_process": server_process}

        root_logger.info("Started!")

    @classmethod
    def close(cls, model_path: str = None):

        if model_path is not None:
            root_logger.info(f"Closing {model_path} worker...")
            cls._workers[model_path]["worker_process"].terminate()
            cls._workers[model_path]["server_process"].terminate()
            cls._workers.pop(model_path)
        else:
            for path in list(cls._workers.keys()):
                cls.close(path)

            cls.controller_process.terminate()
            cls.controller_process = None


def set_seed(seed: int):
    global GLOBAL_SEED
    print(f"Setting random seed to {seed}")
    GLOBAL_SEED = seed
    transformers.set_seed(GLOBAL_SEED)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)


def model_info_from_name(target_model_name: str, fastchat_enabled=True) -> tuple[str, ModelSrc, PreTrainedModel | None, PreTrainedTokenizer | None]:
    if target_model_name.startswith("dev/"):
        model_name, model_src, model_class, tokenizer_class = target_model_name, ModelSrc.DEV, None, None
    elif target_model_name.startswith("meta-llama/") or target_model_name.startswith("mistralai/"):
        model_name, model_src, model_class, tokenizer_class = target_model_name, ModelSrc.OPENAI_API, None, None
        if fastchat_enabled:
            FastChatController.open(model_name)
    elif "gpt-" in target_model_name:
        model_name, model_src, model_class, tokenizer_class = target_model_name, ModelSrc.OPENAI_API, None, None
    else:
        raise ValueError(f"Unknown model name '{target_model_name}'")

    return model_name, model_src, model_class, tokenizer_class
