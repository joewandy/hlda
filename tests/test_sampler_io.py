import os
import sys
import csv
from importlib import import_module

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

sampler = import_module("hlda.sampler")


def test_load_vocab(tmp_path):
    vocab_file = tmp_path / "vocab.csv"
    with open(vocab_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([0, " hello"])
        writer.writerow([1, " world "])

    vocab = sampler.load_vocab(str(vocab_file))

    assert vocab == ["hello", "world"]


def test_load_corpus(tmp_path):
    corpus_file = tmp_path / "corpus.csv"
    with open(corpus_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["0 hello", "1 world"])
        writer.writerow(["1 world", "0 hello "])

    corpus = sampler.load_corpus(str(corpus_file))

    assert corpus == [[0, 1], [1, 0]]
