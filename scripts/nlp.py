import os
import time
import argparse
import pandas as pd
import random
from benchmark_tasks import nlp_tasks
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


def main(args):
    start_time = time.time()
    task_name = args.task
    prompt_template = nlp_tasks[task_name]
    run_results = []
    output_filename = "outputs/run_results_%s_%s_%s.csv" % (
        task_name,
        args.model.replace("/", "_"),
        time.strftime("%Y%m%d-%H%M%S"),
    )
    batch_size = args.batch_size
    data_dir = os.path.join("data", "canto-nlp")
    model = None

    if "sentiment" in task_name:
        max_tokens = 4
        random_output = ["Positive", "Negative", "Neutral"]

    else:
        max_tokens = 1024
        random_output = ["Dummy output 1", "Dummy output 2"]

    if "gemini" in args.model:
        assert args.api_key is not None, "API key is required for Google vertex AI"
        model = VertexAIInference(
            args.model,
            args.api_key,
            batch_size=1,
            temperature=0.0,
            max_tokens=max_tokens,
            stop_sequences=[],
        )
    elif "claude" in args.model:
        assert args.api_key is not None, "API key is required for Anthropic"
        model = AnthropicInference(
            args.model,
            args.api_key,
            batch_size=1,
            temperature=0.0,
            max_tokens=max_tokens,
            stop_sequences=[],
        )
    elif "ERNIE" in args.model:
        assert args.api_key is not None, "API key is required for Baidu"
        model = BaiduInference(
            args.model,
            args.api_key,
            batch_size=1,
            temperature=0.01,
            max_tokens=max_tokens,
            stop_sequences=[],
        )
    elif "ep-" in args.model:
        assert args.api_key is not None, "API key is required for xFyun"
        model = VolcEngineInference(
            args.model,
            args.api_key,
            batch_size=1,
            temperature=0.0,
            max_tokens=max_tokens,
            stop_sequences=[],
        )
    elif (
        "gpt" in args.model
        or "siliconflow/" in args.model  # or args.model == "4.0Ultra"
        or "deepseek" in args.model
        or args.model_url is not None
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
            max_tokens=max_tokens,
            stop_sequences=[],
        )
    elif "/" in args.model:  # local / HF models
        model = LocalInference(
            args.model,
            batch_size=batch_size,
            temperature=0.1,
            max_tokens=max_tokens,
            stop_sequences=[],
            use_chat_template=True,
        )
    elif args.model == "random":
        model = RandomInference(
            random_output,
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

    if "translation" in task_name:
        is_line_file = True
        text_col_name = "SourceText"
        src_lang = task_name.replace("fewshot_", "").split("_")[0]
        tgt_lang = task_name.replace("fewshot_", "").split("_")[1]

        print(data_dir, f"{src_lang}_{tgt_lang}_train.json")

        if "fewshot" in task_name:
            dev_df = pd.read_json(
                os.path.join(data_dir, f"{src_lang}_{tgt_lang}_train.json"),
                lines=is_line_file,
            )
            fewshot_sample_df = dev_df.sample(3)
            example_pairs = []

            for i in range(3):
                example_pairs.append(
                    (
                        fewshot_sample_df.iloc[i].SourceText,
                        fewshot_sample_df.iloc[i].TargetText,
                    )
                )
            prompt_template = prompt_template.format(
                src_example1=example_pairs[0][0],
                tgt_example1=example_pairs[0][1],
                src_example2=example_pairs[1][0],
                tgt_example2=example_pairs[1][1],
                src_example3=example_pairs[2][0],
                tgt_example3=example_pairs[2][1],
            )
        test_filename = f"{src_lang}_{tgt_lang}_test.json"

    if "summarization" in task_name or "translation" in task_name:

        test_df = pd.read_json(
            os.path.join(data_dir, test_filename), lines=is_line_file
        )
        test_df.rename(columns={0: "text"}, inplace=True)
        inputs = [row[text_col_name] for i, row in test_df.iterrows()]
        responses = model.batch_infer([prompt_template.format(text) for text in inputs])
        result = [
            {
                "input": input,
                "response": response,
            }
            for input, response in zip(inputs, responses)
        ]
        run_results.extend(result)

        pd.DataFrame(run_results).to_csv(output_filename, index=False)

        end_time = time.time()
        print("total run time %.2f" % (end_time - start_time))

    if "sentiment" in task_name:
        from benchmark_tasks import SENTIMENT_TASKS
        import json

        run_results = {}
        for sentiment_task in SENTIMENT_TASKS:
            test_filename = f"sentiment_{sentiment_task}.json"
            test_df = pd.read_json(os.path.join(data_dir, test_filename), lines=True)

            prompt_template_task = prompt_template.format(
                sentiment_example1=test_df.iloc[0].text,
                sentiment_target1=test_df.iloc[0].sentiment,
            )

            test_df = test_df.iloc[1:]

            inputs = [row[text_col_name] for i, row in test_df.iterrows()]
            responses = model.batch_infer(
                [prompt_template_task.format(text) for text in inputs]
            )

            # for text in inputs:
            #     print(prompt_template.format(text))

            result = [b for b in responses]

            gold_answers = [row.sentiment for i, row in test_df.iterrows()]

            run_results[sentiment_task] = {
                "pred_answers": result,
                "gold_answers": gold_answers,
            }

        with open(output_filename.replace(".csv", ".json"), "w") as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)

        output_acc_filename = output_filename.replace(".csv", "_result.txt")
        with open(output_acc_filename, "w") as f_out:
            with open(output_filename.replace(".csv", ".json"), "r") as f:
                run_results = json.load(f)
            total_acc = 0
            total_num = 0

            for task in SENTIMENT_TASKS:
                acc = 0
                pred_answers = run_results[task]["pred_answers"]
                gold_answers = run_results[task]["gold_answers"]
                for pred, gold in zip(pred_answers, gold_answers):
                    if gold in pred:
                        acc += 1
                line = "%s: %.4f" % (task, acc / len(gold_answers))
                print(line)
                f_out.write(line + "\n")
                total_acc += acc
                total_num += len(gold_answers)
            line = "all: %.4f" % (total_acc / total_num)
            print(line)
            f_out.write(line + "\n")
            f_out.close()

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
        "--task", type=str, default="summarization", choices=nlp_tasks.keys()
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
