import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    retry_if_not_exception_type,
)
from .inference import Inference

from google import genai
from google.genai import types

wait_times = (
    [wait_fixed(3) for i in range(3)]
    + [wait_fixed(5) for i in range(2)]
    + [wait_fixed(10)]
)

class GeminiInference(Inference):
    def __init__(
        self,
        model_name=str,
        api_key=str,
        batch_size=1,
        system_prompt=None,
        stop_sequences=["\\n"],
        temperature=0.0,
        max_tokens=5,
    ):
        super().__init__(
            model_name,
            batch_size,
            system_prompt,
            stop_sequences,
            temperature,
            max_tokens,
        )
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)

    @retry(stop=stop_after_attempt(8), wait=wait_chain(*wait_times))
    def infer(self, prompt: str):


        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]


        if self.system_prompt:
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                system_instruction=[
                    types.Part.from_text(text=self.system_prompt),
                ],
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                stopSequences=self.stop_sequences,
            )

        else:
            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                stopSequences=self.stop_sequences,
            )


        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generate_content_config,
        )

        return response.text