import os
import time
import argparse
import pandas as pd
from benchmark_tasks import benchmark_tasks
import json
from models import (
    LocalInference,
    VertexAIInference,
    AnthropicInference,
    OpenAIInference,
    RandomInference,
    BaiduInference,
    VolcEngineInference,
)


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s.strip()


def format_example(df, idx, choices, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, choices, k=-1):
    extra_prompt = ""

    if k == 0:
        extra_prompt = " DO NOT EXPLAIN."

    if subject == "local_knowledge":
        subject = "geography"

    prompt = "The following are multiple choice questions (with answers) about {}.{}\n\n".format(
        format_subject(subject), extra_prompt
    )

    if k == -1:
        k = train_df.shape[0]

    for i in range(k):
        prompt += format_example(train_df, i, choices)
    return prompt


def compute_metric(output_filename):
    output_acc_filename = output_filename.replace(".json", "_result.txt")

    print(output_filename, output_acc_filename)

    with open(output_acc_filename, "w") as f_out:
        with open(output_filename, "r") as f_int:
            run_results = json.load(f_int)
        total_acc = 0
        total_num = 0
        for task in run_results:
            acc = 0
            pred_answers = run_results[task]["pred_answers"]
            gold_answers = run_results[task]["gold_answers"]
            for pred, gold in zip(pred_answers, gold_answers):
                if len(pred) > 0 and pred[0].upper() == gold.upper():
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


def main(args):
    start_time = time.time()
    task_name = args.task
    tasks, choices, system_prompt = benchmark_tasks[task_name]
    run_results = {}
    output_filename = "outputs/run_results_%s_%s_%s.json" % (
        task_name,
        args.model.replace("/", "_"),
        time.strftime("%Y%m%d-%H%M%S"),
    )
    batch_size = args.batch_size
    data_dir = os.path.join("data", task_name)
    model = None

    if "gemini" in args.model:
        assert args.api_key is not None, "API key is required for Google vertex AI"
        model = VertexAIInference(
            args.model,
            args.api_key,
            batch_size=1,
            system_prompt=system_prompt,
            temperature=0.0,
        )
    elif "claude" in args.model:
        assert args.api_key is not None, "API key is required for Anthropic"
        model = AnthropicInference(
            args.model,
            args.api_key,
            batch_size=1,
            system_prompt=system_prompt,
            temperature=0.0,
        )
    elif "ERNIE" in args.model:
        assert args.api_key is not None, "API key is required for Baidu"
        model = BaiduInference(
            args.model,
            args.api_key,
            batch_size=1,
            system_prompt=system_prompt,
            temperature=0.01,
        )
    elif "ep-" in args.model:
        # assert args.api_key is not None, "API key is required for Doubao"
        model = VolcEngineInference(
            args.model,
            args.api_key,
            batch_size=1,
            system_prompt=system_prompt,
            temperature=0.0,
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
            system_prompt=system_prompt,
            temperature=0.0,
        )
    elif "/" in args.model:  # local / HF models
        model = LocalInference(
            args.model,
            batch_size=batch_size,
            system_prompt=system_prompt,
            temperature=0.1,
            load_in_8bit=args.load_in_8bit,
            dtype=args.dtype,
        )
    elif args.model == "random":
        model = RandomInference(
            choices,
            args.model,
            batch_size=batch_size,
            system_prompt=system_prompt,
            temperature=0.0,
        )

    if model is None:
        raise ValueError("model not supported")

    if "mmlu" in task_name:
        import glob
        from os.path import basename, splitext

        if os.path.exists(
            f"outputs/checkpoint_{task_name}_{args.model.replace('/', '_')}"
        ):

            files = glob.glob(
                f"outputs/checkpoint_{task_name}_{args.model.replace('/', '_')}/*.json"
            )

            if len(files) == 0:
                mmlu_task_id = 0

            elif len(files) == len(tasks):
                print("All tasks have been tested")
                return

            else:
                files = sorted(
                    files, key=lambda x: int(splitext(basename(x))[0].split("_")[0])
                )
                last_file = splitext(basename(files[-1]))[0]
                last_task = int(last_file.split("_")[0])

                # tasks = tasks[last_task+1:]

                for i, file in enumerate(files):
                    task = tasks[i]
                    with open(file, "r") as f:
                        run_result_task = json.load(f)

                    run_results[task] = run_result_task

                tasks = tasks[last_task + 1 :]
                mmlu_task_id = last_task + 1
                print(f"Continue testing from task {tasks[0]}")

        else:
            os.mkdir(f"outputs/checkpoint_{task_name}_{args.model.replace('/', '_')}")
            mmlu_task_id = 0

    for task in tasks:
        print("Testing %s ..." % task)
        records = []
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", task + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", task + "_test.csv"), header=None
        )

        for i in range(test_df.shape[0]):
            # The following are multiple choice questions (with answers) about
            # get prompt and make sure it fits
            k = args.ntrain
            prompt_end = format_example(test_df, i, choices, include_answer=False)
            train_prompt = gen_prompt(dev_df, task, choices, k)
            prompt = train_prompt + prompt_end
            label = test_df.iloc[i, test_df.shape[1] - 1]
            records.append({"prompt": prompt, "answer": label})

            # break

        inputs = [record["prompt"] for record in records]
        pred_answers = model.batch_infer(inputs)
        pred_answers = [
            # b[0] if len(b) > 0 else "" # only get first character as answer
            b
            for b in pred_answers
        ]
        gold_answers = [record["answer"] for record in records]
        run_results[task] = {"pred_answers": pred_answers, "gold_answers": gold_answers}

        if "mmlu" in task_name:
            with open(
                f"outputs/checkpoint_{task_name}_{args.model.replace('/', '_')}/{mmlu_task_id}_{task}.json",
                "w",
            ) as f:
                json.dump(run_results[task], f, ensure_ascii=False, indent=2)
            mmlu_task_id += 1

    with open(output_filename, "w") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    compute_metric(output_filename)
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
    "Qwen/Qwen2-72B-Instruct",  # Local API
]

model_types = [
    "llama",
    "mistral",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=False)
    parser.add_argument(
        "--task", type=str, default="mmlu", choices=benchmark_tasks.keys()
    )
    parser.add_argument(
        "--model", type=str, default="gemini-1.5-flash-001", required=True
    )
    parser.add_argument("--model_url", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--load_in_8bit", type=bool, default=False)
    args = parser.parse_args()

    main(args)
