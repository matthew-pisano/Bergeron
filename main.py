import datetime
import os
import random
import time

import openai
from dotenv import load_dotenv

from src.models.model_utils import ModelSrc
from src.models.primary import Primary

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    main_start = time.time()
    print(f"Begin main at {datetime.datetime.utcfromtimestamp(main_start)} UTC")

    # model_name, model_src = "gpt-3.5-turbo", ModelSrc.OPENAI_API
    # model_name, model_src = "PY007/TinyLlama-1.1B-step-50K-105b", ModelSrc.LOCAL
    model_name, model_src = "Open-Orca/OpenOrca-Platypus2-13B", ModelSrc.HF_API
    # model_name, model_src = "gpt2", ModelSrc.HF_API

    primary = Primary(model_name, model_src)

    result = primary.generate("summarize: How much wood could a woodchuck chuck")

    print(result)

    main_end = time.time()
    print(f"End main at {datetime.datetime.utcfromtimestamp(main_end)} UTC")
    print(f"Elapsed time of {round(main_end-main_start, 3)}s")


if __name__ == "__main__":
    main()
