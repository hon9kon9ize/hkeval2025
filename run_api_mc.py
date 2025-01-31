import argparse
import subprocess

tasks = [
    "canto-mmlu",
    "cultural",
    "dse",
    "hk-law",
    "mmlu",
    "phonetics",
    "professional",
]


def main(model: str, api_key: str, model_url: str = None):
    for task in tasks:
        print("---")
        print("Run task:", task)
        print("Model:", model)
        cmd = f"python3.11 scripts/eval.py --task {task} --model=\"{model}\" --api_key {api_key} --batch_size 1"

        if model_url is not None:
            cmd += f" --model_url {model_url}"

        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--model_url", type=str, default=None)
    args = parser.parse_args()

    main(args.model, args.api_key, args.model_url)
