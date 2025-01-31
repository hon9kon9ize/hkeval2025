from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
from .inference import Inference
import qianfan

wait_times = (
    [wait_fixed(3) for i in range(3)]
    + [wait_fixed(5) for i in range(2)]
    + [wait_fixed(10)]
)


class BaiduInference(Inference):
    def __init__(
        self,
        model_name=str,
        api_key=str,
        batch_size=1,
        system_prompt=None,
        stop_sequences=["\\n"],
        temperature=0.0,
        max_tokens=2,
    ):
        super().__init__(
            model_name,
            batch_size,
            system_prompt,
            stop_sequences,
            temperature,
            max_tokens,
        )
        access_key_id, secret_access_key = api_key.split(':')
        qianfan.AccessKey(access_key_id)
        qianfan.SecretKey(secret_access_key)
        self.chat_comp = qianfan.ChatCompletion()

    @retry(stop=stop_after_attempt(8), wait=wait_chain(*wait_times))
    def infer(self, prompt: str):
        response = self.chat_comp.do(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            system=self.system_prompt,
            max_output_tokens=self.max_tokens,
            stop=self.stop_sequences,
            temperature=self.temperature,
        )
        response = response['body']['result']

        return response
