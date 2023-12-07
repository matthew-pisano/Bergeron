import random
import string
from typing import Union

from datasets import load_dataset


class Benchmark:

    repo = ""
    test_split = ""
    dev_split = ""

    def __init__(self, config_name="all", split: list[str] | None = None):

        if split is None:
            split = [self.test_split]
        if self.dev_split not in split:
            split.append(self.dev_split)

        split_dataset = load_dataset(self.repo, config_name, split=split)
        self.dataset = {split_name: split_dataset[split_idx] for split_idx, split_name in enumerate(split)}
        self.split = split

    def split_prompts(self, split_name: str = ""):

        if split_name == "":
            split_name = self.split[0]

        return self.dataset[split_name]

    def format_question(self, question_id: int, split_name: str = "", n_shot=0, as_example=False) -> str:
        ...

    def batch_format_questions(self, split_name: str = "", n_shot=0):
        num_prompts = len(self.split_prompts(split_name))
        return [self.format_question(i, split_name=split_name, n_shot=n_shot, as_example=False) for i in range(num_prompts)]

    def evaluate_answer(self, question_id: int, model_response: str, split_name: str = ""):
        ...

    def batch_evaluate_answers(self, model_responses: list[str], split_name: str = ""):

        return [self.evaluate_answer(i, model_responses[i], split_name=split_name) for i in range(len(model_responses))]


class MMLU(Benchmark):

    repo = "cais/mmlu"
    test_split = "test"
    dev_split = "dev"

    def format_question(self, question_id: int, split_name: str = "", n_shot=0, as_example=False) -> str:

        examples = []
        num_dev = len(self.dataset["dev"])
        while len(examples) < n_shot:
            examples.append(self.format_question(random.randint(0, num_dev-1), split_name="dev", n_shot=0, as_example=True))

        sample = self.split_prompts(split_name=split_name)[question_id]
        choices = "\n".join([f"{string.ascii_lowercase[i]}) {choice}" for i, choice in enumerate(sample["choices"])])
        question = f"""{sample["question"]}
{choices}
Answer: """

        if as_example:
            question += string.ascii_lowercase[sample["answer"]]

        prompt = "" if as_example else "Please answer the following question to the best of your ability. In your output give the answer ONLY. Do not give any explanation or other text. Just the letter choice if your answer."
        if len(examples) > 0:
            prompt += "\nFor example:\n"+"\n\n".join(examples)
        prompt += ("\n\nNow, answer this question ONLY, nothing else.\n" if not as_example else "")+question

        return prompt

    def evaluate_answer(self, question_id: int, model_response: str, split_name: str = ""):

        answer = string.ascii_lowercase[self.split_prompts(split_name=split_name)[question_id]["answer"]]
        model_response = model_response.replace(")", " ").strip()

        return model_response[0] == answer


def benchmark_from_name(benchmark_name: str, config_name: str = None, split: list[str] | None = None) -> Benchmark:

    if "mmlu" in benchmark_name.lower():
        return MMLU(config_name=config_name, split=split)

    raise ValueError("Could not find benchmark "+benchmark_name)


if __name__ == "__main__":
    mmlu = MMLU()
    for i in range(10):
        print("\n<<--<question>-->>\n")
        question = mmlu.format_question(i, n_shot=3)
        print(question)

