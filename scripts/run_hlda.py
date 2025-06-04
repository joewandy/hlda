#!/usr/bin/env python3
"""Command-line utility for running hierarchical LDA on a corpus of text
files."""

import argparse
import glob
import os
import re


from hlda.sampler import HierarchicalLDA

# A small set of English stopwords. This keeps the demo self-contained.
STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "with",
    "that",
    "this",
    "from",
    "you",
    "was",
    "have",
    "not",
    "but",
    "they",
    "his",
    "her",
    "she",
    "has",
    "had",
    "him",
    "its",
    "our",
    "their",
    "about",
    "into",
    "after",
    "these",
    "those",
    "them",
    "over",
    "such",
    "also",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "your",
    "than",
    "when",
    "where",
    "what",
    "which",
    "who",
    "whom",
    "why",
    "how",
}

TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")


def load_documents(data_dir: str):
    """Load and preprocess all text files under *data_dir*."""
    corpus = []
    for filename in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().lower()
        tokens = [t for t in TOKEN_RE.findall(text) if t not in STOPWORDS]
        corpus.append(tokens)
    return corpus


def build_vocab(corpus):
    vocab = sorted({word for doc in corpus for word in doc})
    index = {w: i for i, w in enumerate(vocab)}
    return vocab, index


def convert_corpus(corpus, index):
    new_corpus = []
    for doc in corpus:
        new_corpus.append([index[w] for w in doc])
    return new_corpus


def run_demo(args):
    corpus = load_documents(args.data_dir)
    vocab, index = build_vocab(corpus)
    int_corpus = convert_corpus(corpus, index)

    hlda = HierarchicalLDA(
        int_corpus,
        vocab,
        alpha=args.alpha,
        gamma=args.gamma,
        eta=args.eta,
        num_levels=args.num_levels,
        seed=args.seed,
    )

    hlda.estimate(
        args.iterations,
        display_topics=args.display_topics,
        n_words=args.n_words,
        with_weights=False,
    )

    print("\nFinal topic hierarchy:")
    hlda.print_nodes(args.n_words, with_weights=False)

    return hlda


def main():
    parser = argparse.ArgumentParser(
        description="Run hierarchical LDA on a directory of text documents"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing text files"
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
