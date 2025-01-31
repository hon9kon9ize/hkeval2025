from typing import List
from tqdm.auto import tqdm
from abc import ABC, abstractmethod


class Inference(ABC):
    def __init__(
        self,
        model_name: str,
        batch_size=1,
        system_prompt=None,
        stop_sequences=["\\n"],
        temperature=0.0,
        max_tokens=1,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.system_prompt = system_prompt
        self.stop_sequences = stop_sequences
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def infer(self, prompt: str) -> str:
        pass

    def batch_infer(self, prompts: List[str]) -> List[str]:
        batch_prompts = self.batch_split(prompts)
        answers = []

        for mini_batch in tqdm(batch_prompts, leave=False):
            answers.extend([self.infer(prompt) for prompt in mini_batch])

        return answers

    def batch_split(self, prompts: List[str]):
        batch_prompts = []
        mini_batch = []

        for prompt in prompts:
            mini_batch.append(prompt)
            if len(mini_batch) == self.batch_size:
                batch_prompts.append(mini_batch)
                mini_batch = []

        if len(mini_batch) != 0:
            batch_prompts.append(mini_batch)

        return batch_prompts
