
import numpy as np
import pytest

from document_intelligence_ml.vectorization import create_vectorizer


@pytest.fixture
def documents():
    return [
        "Apple Announces New Phone",
        "Apple announces new tablet",
        "Microsoft Releases New Software",
        "Microsoft releases new update",
    ]


def test_text_normalization(documents):
    vectorizer = create_vectorizer(min_df=1, max_df=1.0)
    vectorizer.fit(documents)

    vocabulary = vectorizer.get_feature_names_out()

    assert "apple" in vocabulary
    assert "Apple" not in vocabulary


def test_matrix_shape(documents):
    vectorizer = create_vectorizer()
    features = vectorizer.fit_transform(documents)

    assert features.shape[0] == len(documents)
    assert features.shape[1] <= 20_000


def test_float32_output(documents):
    vectorizer = create_vectorizer()
    features = vectorizer.fit_transform(documents)

    assert features.dtype == np.float32


def test_bigram_extraction(documents):
    vectorizer = create_vectorizer()
    vectorizer.fit(documents)

    vocabulary = vectorizer.get_feature_names_out()

    assert "apple announces" in vocabulary
    assert "microsoft releases" in vocabulary


def test_rare_word_filtering(documents):
    vectorizer = create_vectorizer()
    vectorizer.fit(documents)

    vocabulary = vectorizer.get_feature_names_out()

    assert "phone" not in vocabulary
    assert "tablet" not in vocabulary


def test_common_word_filtering(documents):
    vectorizer = create_vectorizer(max_df=0.75)
    vectorizer.fit(documents)

    vocabulary = vectorizer.get_feature_names_out()

    assert "new" not in vocabulary


def test_unseen_document(documents):
    vectorizer = create_vectorizer()
    train_features = vectorizer.fit_transform(documents)
    vocabulary_before = vectorizer.vocabulary_.copy()

    new_features = vectorizer.transform([
        "Apple announces new technology"
    ])

    assert new_features.shape == (1, train_features.shape[1])
    assert vectorizer.vocabulary_ == vocabulary_before


def test_l2_normalization(documents):
    vectorizer = create_vectorizer()
    features = vectorizer.fit_transform(documents)

    norms = np.sqrt(features.multiply(features).sum(axis=1))
    np.testing.assert_allclose(
        np.asarray(norms).ravel(),
        np.ones(len(documents)),
        rtol=1e-5,
    )