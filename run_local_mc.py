import argparse
import subprocess

model_list = [
    # "random",
    # "google/gemma-2-2b-it",
    # "google/gemma-2-9b-it",
    # "google/gemma-2-27b-it",
    # "01-ai/Yi-1.5-6B-Chat",
    # "01-ai/Yi-1.5-9B-Chat",
    # "01-ai/Yi-1.5-34B-Chat",
    # "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "mistralai/Mistral-Nemo-Instruct-2407",
    # "CohereForAI/aya-23-8B",
    # "/home/pj24001684/ku40000295/jc/models/CantonesellmChat-v0.5-34B-sft",
    "/home/pj24001684/ku40000295/jc/models/CantonesellmChat-v0.5-sft2",
    "Qwen/Qwen2-72B-Instruct",
    # "/home/pj24001684/ku40000295/jc/models/Qwen72B-sft/",
]


tasks = [
    "canto-mmlu",
    "mmlu",
    "cultural",
    "dse",
    "hk-law",
    "phonetics",
    "professional",
]


def main(i: int, bs: int, oneshot=False, zeroshot=False):
    model = model_list[i]
    for task in tasks:
        print("---")
        print("Run task:", task)
        print("Model:", model)
        cmd = f"python3.11 scripts/eval.py --task {task} --model {model} --batch_size {bs} --dtype bfloat16"

        if oneshot:
            cmd += " --ntrain 1"

        elif zeroshot:
            cmd += " --ntrain 0"
        cmd += "--load_in_8bit"

        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--oneshot", action="store_true", default=False)
    parser.add_argument("--zeroshot", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    i = args.i
    oneshot = args.oneshot
    zeroshot = args.zeroshot
    bs = args.batch_size

    main(i, bs, oneshot, zeroshot)
