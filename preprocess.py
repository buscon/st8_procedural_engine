#!/usr/bin/env python
"""
python preprocess.py --config config.yaml
"""
import argparse
import yaml
from engine_ddsp.data import run_preprocessing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = run_preprocessing(config)
    print(f"wrote preprocessed dataset to {out_dir}")


if __name__ == "__main__":
    main()
