import argparse
from types import SimpleNamespace

import yaml

from src.evaluator import evaluate


def main(cfg):
    evaluate(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    main(SimpleNamespace(**cfg))
    print("Evaluation finished.")
