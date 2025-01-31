from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    retry_if_not_exception_type,
)
from .inference import Inference
from anthropic import Anthropic, BadRequestError

wait_times = (
    [wait_fixed(3) for i in range(3)]
    # + [wait_fixed(5) for i in range(2)]
    # + [wait_fixed(5) for i in range(2)]
)


# class AnthropicInference(Inference):
#     def __init__(
#         self,
#         model_name=str,
#         api_key=str,
#         batch_size=1,
#         system_prompt=[],
#         stop_sequences=["\\n"],
#         temperature=0.0,
#         max_tokens=1,
#     ):
#         super().__init__(
#             model_name,
#             batch_size,
#             system_prompt,
#             stop_sequences,
#             temperature,
#             max_tokens,
#         )
#         self.client = Anthropic(api_key=api_key)

#     @retry(
#         stop=stop_after_attempt(8),
#         wait=wait_chain(*wait_times),
#         retry=retry_if_not_exception_type((BadRequestError,)),
#     )
#     def infer(self, prompt: str):
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt,
#                     }
#                 ],
#             },
#         ]

#         response = self.client.messages.create(
#             model=self.model_name,
#             messages=messages,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#             system=self.system_prompt,
#         )  # stop_sequences=stop_sequences

#         response = response.content[0].text.strip()

#         return response


import boto3
from botocore.exceptions import ClientError

class AnthropicInference(Inference):
    def __init__(
        self,
        model_name=str,
        api_key=str,
        batch_size=1,
        system_prompt=[],
        stop_sequences=["\\n"],
        temperature=0.01,
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
        aws_access_key_id, aws_secret_access_key = api_key.split(":")

        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name="us-east-1",
        )

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_chain(*wait_times),
        retry=retry_if_not_exception_type((ClientError, Exception)),
    )
    def infer(self, prompt: str):
        conversation = [
            {
                "role": "user",
                "content": [{"text": self.system_prompt + prompt}],
            }
        ]

        response = self.client.converse(
            modelId=self.model_name,
            messages=conversation,
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text