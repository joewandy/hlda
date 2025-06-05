import argparse
import json
import os
import sys
from importlib import import_module

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

run_hlda = import_module("scripts.run_hlda")


def test_export_tree(tmp_path):
    (tmp_path / "doc1.txt").write_text("First document about cats.")
    (tmp_path / "doc2.txt").write_text("Second document about dogs.")

    output_file = tmp_path / "tree.json"
    args = argparse.Namespace(
        data_dir=str(tmp_path),
        iterations=1,
        display_topics=1,
        n_words=2,
        num_levels=3,
        alpha=1.0,
        gamma=1.0,
        eta=0.1,
        seed=0,
        export_tree=str(output_file),
    )

    run_hlda.run_hlda(args)
    data = json.loads(output_file.read_text())

    assert data["level"] == 0
    assert isinstance(data["children"], list)
