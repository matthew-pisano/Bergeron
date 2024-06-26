# Project Bergeron

The goal of this project is to create a framework that protects models against both natural language adversarial attacks and its own bias toward mis-alignment.  This is done through the usage of a secondary model that judges the prompts to and responses from that primary model.  This leaves the primary model and the end user less exposed to potential threats.

This can be thought of attaching a conscience to these models to help guide them toward aligned responses.

## Framework Components

The Bergeron framework is composed of two main components: the primary model and the secondary model.

![Framework component configuration](docs/img/alignmentFlow.png)

### Primary Model

Here, the primary model can be any LLM.  It is unchanged with no special fine-tuning.  It receives prompts from the user and modified prompts from the secondary model if needed.

### Secondary Model

The secondary model serves as the primary model's conscience.  It judges incoming prompts as safe or unsafe as well as the outgoing responses from the primary model.  The goal is to:

1. Prevent unsafe prompts from reaching the primary model
2. Prevent unsafe responses from reaching the end user if step 1 fails.

## Quick Start

### Software Requirements

* Python >= 3.10
* A python environment with the modules from the [requirements.txt](requirements.txt) installed
* (Optional but recommended) *FastChat* from LM-SYS installed as a python module (in requirements)
* A set of valid OpenAI and Huggingface API keys to be stored in a `.env` file

## Generative Models

Bergeron can work with any appropriate text-generation model that is available through HuggingFace or OpenAI.  We also use several open-source manual/debugging models for evaluation.  The following can, and some have, been used as either the primary or secondary model in our framework:

* HuggingFace
  * `mistralai/Mistral-7B-Instruct-v0.1`
  * `meta-llama/Llama-2-7b-chat-hf`
* OpenAI
  * `openai/gpt-3.5-turbo`
  * `openai/gpt-4-turbo-preview`
* Manual/Debugging
  * `dev/singleline`
    * A model that prompts the user for a single line of text
  * `dev/multiline`
    * A model that prompts the user for one or more lines of text

## Sandbox Testing

We provide a [sandbox.py](sandbox.py) file to allow for the testing of our model on individual prompts without performing an entire evaluation.

Here, you can try out different framework configurations on your own custom prompts without using any pre-made datasets.  Using both primary and secondary models here will run the full Bergeron framework on your prompt.  Using just the primary model will give the prompt to the chosen model without any modifications or safety checks.

### Single Prompt

This will perform a single inference using the query provided.

```shell
python sandbox.py query --primary <primary_model> --secondary <secondary_model> --prompt <prompt>
```

### Multiple Prompts

This will allow for multiple user inputs without restarting the program.

```shell
python sandbox.py converse --primary <primary_model> --secondary <secondary_model> --prompt <prompt>
```

**NOTE:** Context is *NOT* accumulated over multiple prompts.  Each prompt is evaluated independently of all others.

## Evaluation

We also provide an [evaluate.py](evaluate.py) file to allow for the generation of responses in bulk and the analysis of those responses using multiple methods.

### Response Generation

Before evaluations can begin, responses must be generated.  This command generates a response for all prompts in the given dataset.  Prompts are loaded from either `data/prompts/<dataset>` or HuggingFace and responses are saved to `data/responses/<dataset>`.

```shell
python evaluate.py respond <dataset> --primary <primary_model> --secondary <secondary_model>
```

### Evaluating Response Safety

The first evaluation we perform is judging whether the responses generated by a particular model are safe or not.  These judgments can be generated either manually with `--evaluator dev/singleline` or through an LLM.  Responses are loaded from `data/responses/<dataset>` and evaluations are saved to `data/evaluations/<dataset>`.

```shell
python evaluate.py eval_responses <dataset> --primary <primary_model> --secondary <secondary_model> --evaluator <eval_model>
```

### Evaluating Secondary Detections

The second evaluation we can perform is to see how often the secondary model judged a prompt as unsafe/adversarial.  This can tell us several things including how often it detects adversarial prompts and how often it detects non-adversarial prompts as false positives.  Responses are loaded from `data/responses/<dataset>` and evaluations are saved to `data/evaluations/<dataset>`.

```shell
python evaluate.py eval_reports <dataset> --primary <primary_model> --secondary <secondary_model>
```