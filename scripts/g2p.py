import os
import time
import argparse
import pandas as pd
import random
from benchmark_tasks import phonetics_g2p_tasks
from models import (
    LocalInference,
    VertexAIInference,
    AnthropicInference,
    OpenAIInference,
    RandomInference,
    BaiduInference,
    VolcEngineInference,
)

random.seed(42)



from pprint import pprint
import re
import difflib
from Levenshtein import ratio
from statistics import mean

class AccuracyMetrics:
    sentence: str
    totalsyll: int
    correctsyll: int
    cer: float # Characer Error Rate
    leven: float # Levenshtein distance
    diff: list[str] # Difference between response and answer
    def __str__(self) -> str:
        return f'#{self.sentence_id} - {self.sentence}\nCER: 1 - ({self.correctsyll}/{self.totalsyll}) ({self.cer:.3f}); Levenshtein ratio: {self.leven:.3f}'
    def print_diff(self) -> None:
        pprint(self.diff)


def explode_answer(_answer:str)->list[str]:
    m = re.search(r'\([^\(]*\|[^\(]*\)', _answer)
    answers = []
    if m:
        block = m.group(0)
        for s in [explode_answer(_answer.replace(block, pron ,1)) for pron in block[1:-1].split('|')]:
            answers.extend(s)
    else:
        answers.append(_answer)
    return answers

def jyutping_accuracy(_honzi: str, _response: str, _answer: str) -> AccuracyMetrics:
    result = AccuracyMetrics()
    answers = explode_answer(_answer)
    normalized_response = re.sub(r'[^a-z1-6\s]+',' ',_response.lower())
    normalized_response = re.sub(r'([1-6])[\s]*','\\1 ',normalized_response).strip()
    best_ratio = 0
    selected_answer = ''
    for answer in answers:
        r = ratio(answer, normalized_response)
        if r > best_ratio:
            selected_answer = answer
            best_ratio = r
    result.sentence = _honzi
    result.diff = list(difflib.Differ().compare([selected_answer], [normalized_response]))
    result.totalsyll = len(selected_answer.split(' '))
    result.correctsyll = len([1 for a,b in zip(selected_answer.split(' '), normalized_response.split(' ')) if a == b])
    result.cer = 1 - (result.correctsyll / result.totalsyll)
    result.leven = 1 - best_ratio
    return result






def main(args):
    start_time = time.time()
    task_name = args.task
    prompt_template = phonetics_g2p_tasks[task_name]
    run_results = []
    output_filename = "outputs/run_results_%s_%s_%s.csv" % (
        task_name,
        args.model.replace("/", "_"),
        time.strftime("%Y%m%d-%H%M%S"),
    )
    batch_size = args.batch_size
    data_dir = os.path.join("data", "phonetics-g2p")
    model = None

    if "gemini" in args.model:
        assert args.api_key is not None, "API key is required for Google vertex AI"
        model = VertexAIInference(
            args.model,
            args.api_key,
            batch_size=1,
            temperature=0.0,
            max_tokens=1024,
            stop_sequences=[],
        )
    elif (
        "gpt" in args.model or "siliconflow/" or "deepseek" in args.model
    ):
        assert args.api_key is not None, "API key is required for OpenAI"

        model_name = args.model

        if "siliconflow/" in args.model:
            model_name = args.model.replace("siliconflow/", "")

        model = OpenAIInference(
            model_name,
            args.api_key,
            batch_size=1,
            api_url=args.model_url,
            temperature=0.0,
            max_tokens=1024,
            stop_sequences=[],
        )
    elif "claude" in args.model:
        assert args.api_key is not None, "API key is required for Anthropic"
        model = AnthropicInference(
            args.model,
            args.api_key,
            batch_size=1,
            temperature=0.0,
            max_tokens=1024,
            stop_sequences=[],
        )
    elif "ERNIE" in args.model:
        assert args.api_key is not None, "API key is required for Baidu"
        model = BaiduInference(
            args.model,
            args.api_key,
            batch_size=1,
            temperature=0.01,
            max_tokens=1024,
            stop_sequences=[],
        )
    elif "ep-" in args.model:
        assert args.api_key is not None, "API key is required for Doubao"
        model = VolcEngineInference(
            args.model,
            args.api_key,
            batch_size=1,
            temperature=0.0,
            max_tokens=1024,
            stop_sequences=[],
        )
    elif "/" in args.model:  # local / HF models
        model = LocalInference(
            args.model,
            batch_size=batch_size,
            temperature=0.1,
            max_tokens=1024,
            stop_sequences=[],
            dtype="bf16",
            use_chat_template=True,
        )
    elif args.model == "random":
        model = RandomInference(
            ["Dummy output 1", "Dummy output 2"],
            args.model,
            batch_size=batch_size,
            temperature=0.1,
        )

    if model is None:
        raise ValueError("model not supported")

    print("Testing %s ..." % task_name)

    test_filename = task_name + ".json"
    is_line_file = False
    text_col_name = "text"

    if "zh_to_jyutping" in task_name:
        is_line_file = True

        # print(data_dir, f"dev/zh_to_jyutping_dev.json")

        # if "fewshot" in task_name:
        dev_df = pd.read_json(
            os.path.join(data_dir, f"dev/zh_to_jyutping_dev.json"),
            lines=is_line_file,
        )
        fewshot_sample_df = dev_df.sample(5)
        example_pairs = []

        for i in range(5):
            example_pairs.append(
                (
                    fewshot_sample_df.iloc[i].Honzi,
                    fewshot_sample_df.iloc[i].Jyutping,
                )
            )
        prompt_template = prompt_template.format(
            src_example1=example_pairs[0][0],
            tgt_example1=example_pairs[0][1],
            src_example2=example_pairs[1][0],
            tgt_example2=example_pairs[1][1],
            src_example3=example_pairs[2][0],
            tgt_example3=example_pairs[2][1],
            src_example4=example_pairs[3][0],
            tgt_example4=example_pairs[3][1],
            src_example5=example_pairs[4][0],
            tgt_example5=example_pairs[4][1],
        )
    test_filename = f"test/zh_to_jyutping_test.json"

    # print(prompt_template)
    test_df = pd.read_json(os.path.join(data_dir, test_filename), lines=is_line_file)
    test_df.rename(columns={"Honzi": "text", "Jyutping": "ground_truth"}, inplace=True)
    inputs = [row[text_col_name] for i, row in test_df.iterrows()]
    ground_truth = [row["ground_truth"] for i, row in test_df.iterrows()]
    responses = model.batch_infer([prompt_template.format(text) for text in inputs])
    
    CER = []
    Levenshtein = []
    
    
    for i in range(len(inputs)):
        result = jyutping_accuracy(inputs[i], responses[i], ground_truth[i])
        CER.append(result.cer)
        Levenshtein.append(result.leven)
        
    
    
    
    result = [
        {
            "input": input,
            "response": response,
            "CER": cer,
            "Levenshtein": leven,
        }
        for input, response, cer, leven in zip(inputs, responses, CER, Levenshtein)
    ]

    
    run_results.extend(result)

    pd.DataFrame(run_results).to_csv(output_filename, index=False)





    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))


model_names = [
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "claude-3-5-sonnet-20240620",
    "gpt-4o",
    "gpt-4o-mini",
    "ERNIE-4.0-8K",  # Baidu
    "4.0Ultra",  # xFyun
    "random",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=False)
    parser.add_argument(
        "--task", type=str, default="zh_to_jyutping", choices=phonetics_g2p_tasks.keys()
    )
    parser.add_argument(
        "--model", type=str, default="gemini-1.5-flash-001", required=True
    )
    parser.add_argument("--model_url", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--dtype", type=str, default=None)
    args = parser.parse_args()

    main(args)
