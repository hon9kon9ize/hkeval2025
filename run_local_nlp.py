import argparse
import subprocess

model_list = [
    "random",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "01-ai/Yi-1.5-6B-Chat",
    "01-ai/Yi-1.5-9B-Chat",
    "01-ai/Yi-1.5-34B-Chat",
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "CohereForAI/aya-23-8B",
    "/home/pj24001684/ku40000295/jc/models/CantonesellmChat-v0.5-34B-sft",
    "/home/pj24001684/ku40000295/jc/models/CantonesellmChat-v0.5-sft2",
]


tasks = [
    "summarization",
    "fewshot_eng_yue_translation",
    "fewshot_yue_eng_translation",
    "fewshot_yue_zh_translation",
    "fewshot_zh_yue_translation",
    "eng_yue_translation",
    "yue_eng_translation",
    "yue_zh_translation",
    "zh_yue_translation",
    "sentiment",
]


def main(i: int):
    model = model_list[i]
    for task in tasks:
        print("---")
        print("Run task:", task)
        print("Model:", model)
        cmd = f"python3.11 scripts/nlp.py --task {task} --model {model} --batch_size 1 --dtype bfloat16"

        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, default=0)
    args = parser.parse_args()
    i = args.i

    main(i)
