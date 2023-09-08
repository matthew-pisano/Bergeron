import os

import openai
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.primary import Primary

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def main():

    # primary = Primary("openai/gpt-3.5-turbo")
    primary = Primary("PY007/TinyLlama-1.1B-step-50K-105b")

    result = primary.generate("summarize: How much wood could a woodchuck chuck...")

    print(result)


if __name__ == "__main__":
    main()
