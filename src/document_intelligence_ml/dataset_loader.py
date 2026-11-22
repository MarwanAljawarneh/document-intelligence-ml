from datasets import load_dataset

def load_news_dataset():
    datasets = load_dataset("fancyzhx/ag_news", split="train")

    print(f"Loaded {len(datasets)} articles.")

    return datasets