from src.framework.base_model import BaseModel


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
