import polars as pl

def explore_categories(dataset):
    df = pl.DataFrame({
        "text": dataset["text"],
        "label": dataset["label"],
    })

    category_counts = (
        df.group_by("label")
        .len()
        .sort("label")
    )

    print(category_counts)

    return category_counts

def check_data_quality(dataset):
    df = pl.DataFrame({
        "text": dataset["text"],
        "label": dataset["label"],
    })

    missing_text = df["text"].null_count()
    empty_text = df.filter(
        pl.col("text").str.strip_chars() == ""
    ).height

    duplicates = df["text"].is_duplicated().sum()

    print(f"Missing text: {missing_text}")
    print(f"Empty text: {empty_text}")
    print(f"Duplicate texts: {duplicates}")

def explore_text_lengths(dataset):
    df = pl.DataFrame({
        "text": dataset["text"],
    })

    df = df.with_columns(
        pl.col("text").str.len_chars().alias("text_length")
    )

    stats = df.select(
        pl.col("text_length").min().alias("min_length"),
        pl.col("text_length").max().alias("max_length"),
        pl.col("text_length").mean().round(2).alias("avg_length"),
    )

    print(stats)

    return stats