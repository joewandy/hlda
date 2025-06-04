import os
import sys

import numpy as np

TEST_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(TEST_DIR, ".."))
from hlda.sampler import HierarchicalLDA


def generate_corpus(n_topics, vocab_size, doc_len, n_docs, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    width = vocab_size // n_topics

    word_dists = np.zeros((n_topics, vocab_size))
    for k in range(n_topics):
        start = k * width
        word_dists[k, start:start + width] = 1.0 / width

    vocab = [f"w{i}" for i in range(vocab_size)]
    corpus = []
    for _ in range(n_docs):
        theta = rng.dirichlet([alpha] * n_topics)
        doc = []
        for _ in range(doc_len):
            k = rng.choice(n_topics, p=theta)
            w = rng.choice(vocab_size, p=word_dists[k])
            doc.append(w)
        corpus.append(doc)
    return corpus, vocab


def test_hlda_runs_on_synthetic_data():
    n_topics = 3
    vocab_size = 9
    doc_len = 20
    n_docs = 5
    corpus, vocab = generate_corpus(n_topics, vocab_size, doc_len, n_docs)

    hlda = HierarchicalLDA(corpus, vocab, alpha=1.0, gamma=1.0, eta=1.0, num_levels=3, seed=0, verbose=False)
    hlda.estimate(2, display_topics=2, n_words=3, with_weights=False)

    assert len(hlda.document_leaves) == n_docs
    assert hlda.root_node.customers == n_docs

def test_tree_invariants_during_sampling():
    n_topics = 3
    vocab_size = 9
    doc_len = 20
    n_docs = 5
    corpus, vocab = generate_corpus(n_topics, vocab_size, doc_len, n_docs)

    hlda = HierarchicalLDA(corpus, vocab, alpha=1.0, gamma=1.0, eta=1.0,
                           num_levels=3, seed=0, verbose=False)

    total_nodes_history = []
    root_cust_history = []
    for _ in range(20):
        for d in range(n_docs):
            hlda.sample_path(d)
        for d in range(n_docs):
            hlda.sample_topics(d)
        total_nodes_history.append(hlda.root_node.total_nodes)
        root_cust_history.append(hlda.root_node.customers)
        for leaf in hlda.document_leaves.values():
            assert leaf.level == hlda.num_levels - 1
            node = leaf
            depth = 0
            while node.parent is not None:
                node = node.parent
                depth += 1
            assert depth == hlda.num_levels - 1

    assert all(cust == n_docs for cust in root_cust_history)
    diffs = np.diff(total_nodes_history)
    assert (diffs > 0).any() and (diffs < 0).any()
