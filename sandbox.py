import datetime
import logging
import random
import time

from dotenv import load_dotenv
import argparse

from universalmodels import ModelSrc
from universalmodels.constants import set_seed
from universalmodels.fastchat import FastChatController
from universalmodels.constants import logger as universal_logger

from src.framework.framework_model import FrameworkModel
from src.constants import logger
from src.framework.bergeron import Bergeron
from src.framework.primary import Primary

# Load OpenAI configuration
load_dotenv()


def converse(model: FrameworkModel, do_sample=True, temperature=0.7, max_new_tokens=None, **kwargs):
    """Allows the user to converse with a given model over multiple prompts.  NOTE: Context is NOT accumulated over time.

    Args:
        model: The model to converse with
        do_sample: Whether the model should use the sampling decoding method
        temperature: The temperature of the model
        max_new_tokens: The number of new tokens to generate"""

    print("Enter ':q' to quit loop\nEnter ':w' to write your response to the model\nEnter ':r' to repeat the last non-command response")

    context = ""
    prev_context = ""
    while True:
        while True:
            response = input(">>> ")
            if response == ":q":
                return
            elif response == ":w":
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
            logger.warning("Keyboard interrupt: canceling generation")
            continue

        logger.info(model_response)
        prev_context = context
        context = ""


def test_query(primary_model_name: str, secondary_model_name: str, prompt: str, model_src: ModelSrc = ModelSrc.AUTO,
               do_sample=True, temperature=0.7, max_new_tokens=None, **kwargs):
    """Creates a bergeron model and queries it with the given prompt

    Args:
        primary_model_name: The name of the model to use as the primary
        secondary_model_name: The name of the model to use as the secondary
        prompt: The prompt to give to the model
        model_src: The source to load the models from
        do_sample: Whether the model should use the sampling decoding method
        temperature: The temperature of the model
        max_new_tokens: The number of new tokens to generate"""

    if prompt is None:
        raise ValueError("You must provide a prompt to query the model")

    if secondary_model_name is not None:
        model = Bergeron.from_model_names(primary_model_name, secondary_model_name, primary_model_src=model_src,
                                          secondary_model_src=model_src)
    else:
        model = Primary.from_model_name(primary_model_name, model_src=model_src)

    response = model.generate(prompt, do_sample=do_sample, temperature=temperature, max_new_tokens=max_new_tokens, **kwargs)

    logger.info("Model response:\n\n", response)


def test_converse(primary_model_name: str, secondary_model_name: str, model_src: ModelSrc = ModelSrc.AUTO):
    """Creates a bergeron model and allows the user to converse with it over multiple prompts.  NOTE: Context is NOT accumulated over time.

    Args:
        primary_model_name: The name of the model to use as the primary
        secondary_model_name: The name of the model to use as the secondary
        model_src: The source to load the models from"""

    if secondary_model_name is not None:
        model = Bergeron.from_model_names(primary_model_name, secondary_model_name, primary_model_src=model_src,
                                          secondary_model_src=model_src)
    else:
        model = Primary.from_model_name(primary_model_name, model_src=model_src)

    converse(model)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["query", "converse"], help="The action to perform")
    parser.add_argument("-p", "--primary", help="The name of the primary model in huggingface format like 'meta-llama/Llama-2-7b-chat-hf'", required=True)
    parser.add_argument("-s", "--secondary", help="The name of the secondary model in huggingface format like 'meta-llama/Llama-2-7b-chat-hf'", default=None)
    parser.add_argument('--prompt', help="The prompt to be given when querying a model", default=None)
    parser.add_argument('--src', help=f"The source to load the models from", choices=[src.value for src in ModelSrc], default=ModelSrc.AUTO.value)
    parser.add_argument('--seed', help="The seed for model inference", default=random.randint(0, 100))
    args = parser.parse_args()

    main_start = time.time()
    logger.info(f"Begin main at {datetime.datetime.utcfromtimestamp(main_start)} UTC")

    logger.setLevel(logging.DEBUG)
    universal_logger.setLevel(logging.DEBUG)

    set_seed(int(args.seed))

    model_src = [src for src in ModelSrc if src.value == args.src][0]

    if args.action == "query":
        test_query(args.primary, args.secondary, args.prompt, model_src=model_src)
    elif args.action == "converse":
        test_converse(args.primary, args.secondary, model_src=model_src)

    FastChatController.close()
    main_end = time.time()
    logger.info(f"End main at {datetime.datetime.utcfromtimestamp(main_end)} UTC")
    logger.info(f"Elapsed time of {round(main_end-main_start, 3)}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        FastChatController.close()
        raise e
