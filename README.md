Hierarchical Latent Dirichlet Allocation
----------------------------------------

**Note: this repository should only be used for education purpose. For production use, I'd recommend using https://github.com/bab2min/tomotopy which is more production-ready**

---

Hierarchical Latent Dirichlet Allocation (hLDA) addresses the problem of learning topic
hierarchies from data. The model relies on a non‑parametric prior called the nested
Chinese restaurant process, which allows for arbitrarily large branching factors and
easily accommodates growing data collections. The hLDA model combines this prior with a
likelihood based on a hierarchical variant of Latent Dirichlet Allocation.

The original papers describing the algorithm are:

- [Hierarchical Topic Models and the Nested Chinese Restaurant Process](http://www.cs.columbia.edu/~blei/papers/BleiGriffithsJordanTenenbaum2003.pdf)
- [The Nested Chinese Restaurant Process and Bayesian Nonparametric Inference of Topic Hierarchies](http://cocosci.berkeley.edu/tom/papers/ncrp.pdf)

## Overview

This repository contains a pure Python implementation of the Gibbs sampler for hLDA.
It is intended for experimentation and as a reference implementation. The code follows
the approach used in the original [Mallet](http://mallet.cs.umass.edu/topics.php)
implementation but with a simplified interface and a fixed depth for the tree.

Key features include:

- **Python 3.11+** support with minimal third‑party dependencies.
- A small set of helper scripts and notebooks demonstrating how to run the sampler.
- Utilities for visualising the resulting topic hierarchy.
- Test suite for verifying the sampler on synthetic data and a small BBC corpus.

## Installation

The package can be installed directly from PyPI:

```bash
pip install hlda
```

Alternatively, to develop locally, clone this repository and install it in editable mode:

```bash
git clone https://github.com/joewandy/hlda.git
cd hlda
pip install -e .
pre-commit install
```

## Usage

The easiest way to get started is by using the sample BBC dataset provided in the
`data/` directory. You can run the full demonstration from the command line:

```bash
python scripts/run_bbc_demo.py --data-dir data/bbc/tech --iterations 20
```

A Jupyter notebook with the same workflow is available at
[`notebooks/bbc_test.ipynb`](notebooks/bbc_test.ipynb). The notebook walks through loading the
corpus, running the sampler and inspecting the learned hierarchy.

Within Python you can also construct the sampler directly:

```python
from hlda.sampler import HierarchicalLDA

corpus = [["word", "word", ...], ...]  # list of tokenised documents
vocab = sorted({w for doc in corpus for w in doc})

hlda = HierarchicalLDA(corpus, vocab, alpha=1.0, gamma=1.0, eta=0.1,
                       num_levels=3, seed=0)
hlda.estimate(iterations=50, display_topics=10)
```

## Running the tests

The repository includes a small test suite that checks the sampler on both the BBC
corpus and synthetic data. After installing the development dependencies you can run:

```bash
pytest -q
```

All tests should pass in a few seconds.

## License

This project is licensed under the terms of the MIT license. See
[`LICENSE.txt`](LICENSE.txt) for details.

