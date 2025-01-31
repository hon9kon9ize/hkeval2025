from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    retry_if_not_exception_type,
)
from .inference import Inference
from openai import OpenAI

wait_times = (
    [wait_fixed(3) for i in range(3)]
    + [wait_fixed(5) for i in range(2)]
    + [wait_fixed(10)]
)


class OpenAIInference(Inference):
    def __init__(
        self,
        model_name=str,
        api_key=str,
        batch_size=1,
        api_url=None,
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
        
        self.client = OpenAI(api_key=api_key, base_url=api_url)

    @retry(stop=stop_after_attempt(8), wait=wait_chain(*wait_times))
    def infer(self, prompt: str):
        messages = [
            {"role": "user", "content": prompt},
        ]

        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt}] + messages

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop_sequences,
        )

        return response.choices[0].message.content.strip()

# from openai import AzureOpenAI
# class OpenAIInference(Inference):
#     def __init__(
#         self,
#         model_name=str,
#         api_key=str,
#         batch_size=1,
#         api_url=None,
#         system_prompt=None,
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
#          )
    
#         endpoint = "https://cantonese-llm-sin.openai.azure.com/"
#         self.client = AzureOpenAI(api_key=api_key, 
#                                 azure_endpoint=endpoint, 
#                                 api_version="2024-05-01-preview",)
        
#     @retry(stop=stop_after_attempt(8), wait=wait_chain(*wait_times))
#     def infer(self, prompt):
#         messages = [
#             {"role": "user", "content": prompt},
#         ]

#         if self.system_prompt:
#             messages = [
#                 {"role": "system", "content": self.system_prompt}] + messages
            
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=messages,
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#             stop=self.stop_sequences,
#         )

#         return response.choices[0].message.content.strip()