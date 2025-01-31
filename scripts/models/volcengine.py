from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
from .inference import Inference
from volcenginesdkarkruntime import Ark

wait_times = (
    [wait_fixed(3) for i in range(3)]
    + [wait_fixed(5) for i in range(2)]
    + [wait_fixed(10)]
)


class VolcEngineInference(Inference):
    def __init__(
        self,
        model_name: str,
        api_key: str,
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
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=api_key
        )

    # @retry(stop=stop_after_attempt(8), wait=wait_chain(*wait_times))
    def infer(self, prompt: str):
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]

        if self.system_prompt != None:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            ] + messages

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return resp.choices[0].message.content
