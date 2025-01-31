from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    retry_if_not_exception_type,
)
from .inference import Inference
import sensenova

wait_times = (
    [wait_fixed(3) for i in range(3)]
    + [wait_fixed(5) for i in range(2)]
    + [wait_fixed(10)]
)


class SenseNovaInference(Inference):
    def __init__(
        self,
        model_name=str,
        api_key=str,
        batch_size=1,
        system_prompt=None,
        stop_sequences=["\\n"],
        temperature=0.0,
        max_tokens=1,
    ):
        super().__init__(
            model_name,
            batch_size,
            system_prompt,
            stop_sequences,
            temperature,
            max_tokens,
        )
        access_key_id, secret_access_key = api_key.split(":")
        sensenova.access_key_id = access_key_id
        sensenova.secret_access_key = secret_access_key

    @retry(stop=stop_after_attempt(8), wait=wait_chain(*wait_times))
    def infer(self, prompt: str):
        messages = [
            {"role": "user", "content": prompt},
        ]

        if self.system_prompt is not None:
            messages = [
                {"role": "system", "content": self.system_prompt},
            ] + messages

        resp = sensenova.ChatCompletion.create(
            messages=messages,
            model=self.model_name,
            max_new_tokens=self.max_tokens,
            n=1,
            temperature=self.temperature,
        )

        return resp["data"]["choices"][0]["message"]
