from random import choice
from .inference import Inference
from typing import List


class RandomInference(Inference):
    def __init__(self, choices, model_name: str, batch_size=1, system_prompt=None, stop_sequences=['\\n'], temperature=0., max_tokens=1):
        super().__init__(model_name, batch_size,
                         system_prompt, stop_sequences, temperature, max_tokens)
        self.choices = choices

    def infer(self, _prompt: str):
        return choice(self.choices)

    def batch_infer(self, prompts: List[str]) -> List[str]:
        answers = [choice(self.choices) for _ in prompts]
        return answers
