import argparse
import pdb
from datasets import load_dataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_name', type=str, required=True, help='HuggingFace dataset repo name')
    args = parser.parse_args()

    dataset = load_dataset(
        args.repo_name,
        split="train",
        keep_in_memory=False,
        num_proc=80,
    )
    pdb.set_trace()
if __name__ == "__main__":
    main()