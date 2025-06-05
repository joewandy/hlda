# Sklearn wrapper for HierarchicalLDA

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from .sampler import HierarchicalLDA


def _dtm_to_corpus(dtm: Any) -> List[List[int]]:
    """Convert a document-term matrix into an integer corpus."""
    if sparse.issparse(dtm):
        dtm = dtm.toarray()
    else:
        dtm = np.asarray(dtm)
    corpus: List[List[int]] = []
    for row in dtm:
        doc: List[int] = []
        for idx, count in enumerate(row):
            if count:
                doc.extend([idx] * int(count))
        corpus.append(doc)
    return corpus


class HierarchicalLDAEstimator(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible estimator for :class:`HierarchicalLDA`."""

    def __init__(
        self,
        *,
        alpha: float = 10.0,
        gamma: float = 1.0,
        eta: float = 0.1,
        num_levels: int = 3,
        iterations: int = 100,
        seed: int = 0,
        verbose: bool = False,
        vocab: Sequence[str] | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
        self.num_levels = num_levels
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose
        self.vocab = list(vocab) if vocab is not None else None

    # ------------------------------------------------------------------
    def _prepare_input(self, X: Any) -> Tuple[List[List[int]], Sequence[str]]:
        corpus: List[List[int]]
        vocab: Sequence[str] | None = None

        if isinstance(X, tuple) and len(X) == 2:
            corpus, vocab = X
        elif sparse.issparse(X) or (isinstance(X, np.ndarray) and X.ndim == 2):
            corpus = _dtm_to_corpus(X)
            vocab = self.vocab
        else:
            corpus = X  # assume already integer corpus
            vocab = self.vocab

        if vocab is None:
            raise ValueError("Vocabulary is required to fit the model")
        return corpus, vocab

    # ------------------------------------------------------------------
    def fit(self, X: Any, y: Any | None = None):  # noqa: D401
        corpus, vocab = self._prepare_input(X)
        self.vocab_ = list(vocab)
        self.model_ = HierarchicalLDA(
            corpus,
            self.vocab_,
            alpha=self.alpha,
            gamma=self.gamma,
            eta=self.eta,
            num_levels=self.num_levels,
            seed=self.seed,
            verbose=self.verbose,
        )
        if self.iterations > 0:
            self.model_.estimate(
                self.iterations,
                display_topics=self.iterations + 1,
                n_words=0,
                with_weights=False,
            )
        return self

    # ------------------------------------------------------------------
    def transform(self, X: Any) -> np.ndarray:  # noqa: D401
        if not hasattr(self, "model_"):
            raise RuntimeError("Estimator has not been fitted")
        n_docs = len(self.model_.document_leaves)
        assignments = np.zeros(n_docs, dtype=int)
        for d in range(n_docs):
            leaf = self.model_.document_leaves[d]
            assignments[d] = leaf.node_id
        return assignments
