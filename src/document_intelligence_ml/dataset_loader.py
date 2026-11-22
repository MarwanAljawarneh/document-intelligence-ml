from datasets import load_dataset
import polars as pl

def load_news_dataset():
    datasets = load_dataset("fancyzhx/ag_news", split="train")

    print(f"Loaded {len(datasets)} articles.")

    return datasets

def save_sample_to_csv(dataset, output_path: str, sample_size: int = 100):
    sample = dataset.select(range(min(sample_size, len(dataset))))

    df = pl.DataFrame({
        "text": sample["text"],
        "label": sample["label"],
    })

    df.write_csv(output_path)

    print(f"Saved {len(df)} articles to {output_path}")