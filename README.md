# Document Intelligence ML

A Python-based machine learning project for classifying text documents using natural language processing (NLP).

The project uses the AG News dataset to classify news articles into four categories: **World, Sports, Business, and Science/Technology**.

## Project Overview

The project follows a standard machine learning pipeline:

**Data Ingestion → Data Exploration → Text Preprocessing → TF-IDF → Model Training → Evaluation**

## Tech Stack

- **Language:** Python
- **Data Processing:** Polars, Hugging Face Datasets
- **Machine Learning:** scikit-learn, NumPy
- **Testing:** pytest
- **Environment Management:** uv

## Dataset

[AG News](https://huggingface.co/datasets/fancyzhx/ag_news) contains 120,000 training articles and 7,600 test articles across four categories.

## Project Structure

```text
src/document_intelligence_ml/
├── ingestion.py
├── dataset_loader.py
├── exploration.py
├── preprocessing.py
└── vectorization.py
tests/
```

## Getting Started

```bash
git clone https://github.com/MarwanAljawarneh/document-intelligence-ml.git
cd document-intelligence-ml
uv sync
```

Run tests:

```bash
uv run pytest
```

## Results

Model training and evaluation are in progress. Accuracy, F1-score, and a confusion matrix will be added after evaluation.

## References

- [AG News Dataset](https://huggingface.co/datasets/fancyzhx/ag_news)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NVIDIA cuML TF-IDF Documentation](https://docs.nvidia.com/cuml/latest/api/generated/cuml.feature_extraction.text.TfidfVectorizer/)