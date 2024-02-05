from typing import List
from collections import defaultdict

from datasets import Dataset, DatasetDict, concatenate_datasets as _concatenate_datasets


def concatenate_datasets(datasets: List[Dataset], column_name: str) -> Dataset:

    splits = defaultdict(list)

    for dataset in datasets:
        if "train" not in dataset:
            raise ValueError(f"All datasets must have a 'train' split. {dataset}")

        columns_to_remove = [col for col in dataset["train"].column_names if col != column_name]

        for split in ["train", "test", "validation"]:
            dataset_split = dataset[split]
            dataset_split.remove_columns(columns_to_remove)
            splits[split].append(dataset_split)

    print(splits)

    return DatasetDict({
        "train": _concatenate_datasets(splits["train"][column_name], column_name),
        "test": _concatenate_datasets(splits["test"], column_name),
        "validation": _concatenate_datasets(splits["validation"], column_name)
    }, info=datasets[0].info)
