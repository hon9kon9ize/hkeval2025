import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    StoppingCriteria,
    BitsAndBytesConfig,
    GenerationConfig,
)
from typing import List
from .inference import Inference
from tqdm.auto import tqdm


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class LocalInference(Inference):
    def __init__(
        self,
        model_name: str,
        batch_size=1,
        system_prompt=None,
        stop_sequences=["\\n"],
        whitelist_tokens=None,
        temperature=0.0,
        max_tokens=3,
        dtype=None,
        load_in_8bit=False,
        use_chat_template=False,
    ):
        super().__init__(
            model_name,
            batch_size,
            system_prompt,
            stop_sequences,
            temperature,
            max_tokens,
        )
        dtype = torch.bfloat16 if dtype == "bfloat16" else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generation_config = None
        quantization_config = None
        self.use_chat_template = use_chat_template

        suppress_tokens = []

        if whitelist_tokens is not None:
            special_token_ids = [
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            ]
            whitelist_token_ids = [
                self.tokenizer.convert_tokens_to_ids(t) for t in whitelist_tokens
            ] + special_token_ids

            # suppress all tokens unless they are explicitly allowed
            for i in range(len(self.tokenizer)):
                if i not in whitelist_token_ids:
                    suppress_tokens.append(i)

        self.suppress_tokens = suppress_tokens

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.generation_config = GenerationConfig.from_pretrained(model_name)
        except:
            pass

        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="balanced_low_0",
            quantization_config=quantization_config,
        )
        self.model.eval()

        self.device = self.model.device

        if stop_sequences is not None:
            stop_sequences = self.tokenizer.convert_tokens_to_ids(stop_sequences)

        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_sequences)])

        # TODO: hotfix for Yi chat models that doesn't have correct eos token in generation config
        if "Yi-1.5" in model_name and "Chat" in model_name:
            print("Yi chat model detected, setting eos token to 7")
            self.tokenizer.eos_token_id = 7
            self.generation_config.eos_token_id = 7
            self.model.generation_config.eos_token_id = 7

        if self.suppress_tokens is not None:
            self.model.generation_config.suppress_tokens = self.suppress_tokens
            self.generation_config.suppress_tokens = self.suppress_tokens

    def infer(self, prompt: str):
        if self.use_chat_template and self.generation_config is not None:
            messages = []

            if self.system_prompt is not None:
                messages = [{"role": "system", "content": self.system_prompt}]

            prompt = self.tokenizer.apply_chat_template(
                messages
                + [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            input_ids,
            temperature=max(self.temperature, 0.1),
            max_new_tokens=self.max_tokens,
            do_sample=True,
            suppress_tokens=self.suppress_tokens,
            stopping_criteria=self.stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        output = self.tokenizer.decode(
            output_ids[0, input_ids.shape[-1] :], skip_special_tokens=True
        )

        return output.strip()

    def batch_infer(self, prompts: List[str]) -> List[str]:
        answers = []

        if self.use_chat_template and self.generation_config is not None:
            messages = []

            if self.system_prompt is not None:
                messages = [{"role": "system", "content": self.system_prompt}]

            prompts = [
                self.tokenizer.apply_chat_template(
                    messages
                    + [
                        {
                            "role": "user",
                            "content": p,
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in prompts
            ]

        for batch_input in tqdm(self.batch_split(prompts), leave=False):
            encode_inputs = self.tokenizer.batch_encode_plus(
                batch_input, return_tensors="pt", padding=True
            ).to(self.device)
            max_input_length = encode_inputs["input_ids"].shape[-1]
            # some model's tokenizer has token_type_ids that would makes model.generate throw error
            if "token_type_ids" in encode_inputs:
                del encode_inputs["token_type_ids"]
            outputs = self.model.generate(
                **encode_inputs,
                do_sample=True,
                temperature=max(self.temperature, 0.0),
                max_new_tokens=self.max_tokens,
                stopping_criteria=self.stopping_criteria,
                suppress_tokens=self.suppress_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # for b in range(outputs.shape[0]):
            #     answers.append(
            #         self.tokenizer.decode(
            #             outputs[b, max_input_length:],
            #             skip_special_tokens=True,
            #         ).strip()
            #     )

            decoded = self.tokenizer.batch_decode(
                outputs[:, max_input_length:], skip_special_tokens=True
            )

            for b in range(len(decoded)):
                answers.append(decoded[b].strip())

        return answers
