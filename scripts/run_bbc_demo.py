#!/usr/bin/env python3
"""Run hierarchical LDA on the included BBC tech dataset."""

import argparse
import os

from scripts.run_hlda import run_demo


def main():
    parser = argparse.ArgumentParser(
        description="Run hierarchical LDA on the BBC tech dataset"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "data", "bbc", "tech"
        ),
        help="Directory containing BBC .txt files",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of Gibbs samples",
    )
    parser.add_argument(
        "--display-topics",
        type=int,
        default=50,
        help="Report topics every N iterations",
    )
    parser.add_argument(
        "--n-words",
        type=int,
        default=5,
        help="Number of words to display per topic",
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=3,
        help="Depth of the topic hierarchy",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Alpha hyperparameter",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma hyperparameter",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Eta hyperparameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )

    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
