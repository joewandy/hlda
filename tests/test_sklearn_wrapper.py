import numpy as np
from importlib import import_module
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


HierarchicalLDAEstimator = import_module(
    "hlda.sklearn_wrapper"
).HierarchicalLDAEstimator  # noqa: E501


def _prepare_input(vectorizer):
    def _transform(X):
        if hasattr(X, "toarray"):
            arr = X.toarray()
        else:
            arr = np.asarray(X)
        corpus = []
        for row in arr:
            doc = []
            for idx, count in enumerate(row):
                doc.extend([idx] * int(count))
            corpus.append(doc)
        vocab = list(vectorizer.get_feature_names_out())
        return corpus, vocab

    return _transform


def test_pipeline_fit_transform():
    docs = [
        "apple orange banana",
        "apple orange",
        "banana banana orange",
    ]

    vectorizer = CountVectorizer()
    hlda = HierarchicalLDAEstimator(
        num_levels=2,
        iterations=1,
        seed=0,
        verbose=False,
    )

    pipeline = Pipeline(
        [
            ("vect", vectorizer),
            (
                "prep",
                FunctionTransformer(
                    _prepare_input(vectorizer),
                    validate=False,
                ),
            ),
            ("hlda", hlda),
        ]
    )

    pipeline.fit(docs)
    result = pipeline.transform(docs)

    assert result.shape[0] == len(docs)
    assert isinstance(result[0], (int, np.integer))
