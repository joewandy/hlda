import os
import sys
from importlib import import_module

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

run_hlda = import_module("scripts.run_hlda")


def test_load_documents(tmp_path):
    (tmp_path / "doc1.txt").write_text(
        "This is the first document. Hello world!"
    )
    (tmp_path / "doc2.txt").write_text(
        "Second document: world is big and bright."
    )

    corpus = run_hlda.load_documents(str(tmp_path))
    assert corpus == [
        ["first", "document", "hello", "world"],
        ["second", "document", "world", "big", "bright"],
    ]


def test_build_vocab():
    corpus = [
        ["first", "document", "hello", "world"],
        ["second", "document", "world", "big", "bright"],
    ]

    vocab, index = run_hlda.build_vocab(corpus)
    expected_vocab = [
        "big",
        "bright",
        "document",
        "first",
        "hello",
        "second",
        "world",
    ]
    expected_index = {w: i for i, w in enumerate(expected_vocab)}
    assert vocab == expected_vocab
    assert index == expected_index


def test_convert_corpus():
    corpus = [
        ["first", "document", "hello", "world"],
        ["second", "document", "world", "big", "bright"],
    ]

    vocab, index = run_hlda.build_vocab(corpus)
    int_corpus = run_hlda.convert_corpus(corpus, index)
    expected = [
        [index[w] for w in corpus[0]],
        [index[w] for w in corpus[1]],
    ]
    assert int_corpus == expected
