import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from scripts import run_hlda

BBC_DIR = os.path.join(ROOT, 'data', 'bbc', 'tech')


def test_bbc_demo_deterministic():
    args = argparse.Namespace(
        data_dir=BBC_DIR,
        iterations=2,
        display_topics=2,
        n_words=3,
        num_levels=3,
        alpha=10.0,
        gamma=1.0,
        eta=0.1,
        seed=0,
    )
    hlda = run_hlda.run_demo(args)
    assert hlda.root_node.total_nodes == 15
    assert hlda.root_node.customers == 401
    assert hlda.num_documents == 401

