
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from document_intelligence_ml.preprocessing import clean_text


# Reference: NVIDIA cuML TfidfVectorizer API
# https://docs.nvidia.com/cuml/latest/api/generated/cuml.feature_extraction.text.TfidfVectorizer/
# Configuration inspired by cuML; implemented with scikit-learn for CPU execution.


def create_vectorizer(
    max_features: int = 20_000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Configure TF-IDF feature extraction for news classification."""

    return TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="word",
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=sublinear_tf,
        norm="l2",
        dtype=np.float32,
    )