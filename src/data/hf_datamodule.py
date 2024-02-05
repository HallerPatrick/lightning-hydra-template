from typing import Any, Dict, Optional, List

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from transformers import AutoTokenizer

from .utils import concatenate_datasets


class HFDataset(Dataset):

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class HFDataModuleForCausalLM(LightningDataModule):

    def __init__(
        self,
        datasets: List[Dict[str, str]],
        tokenizer_name: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        preprocess_procs: int = 1
    ) -> None:
        super().__init__()

        self.raw_datasets = []

        for dataset in datasets:
            raw_dataset = load_dataset(dataset["dataset_name"], dataset["dataset_config_name"])
            assert dataset["column_name"] in raw_dataset["train"].features, \
                f"Column {dataset['column_name']} not found in dataset {dataset['dataset_name']}"
            self.raw_datasets.append({
                "raw_dataset": raw_dataset,
                "column_name": dataset["column_name"]
            })

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.preprocess_procs = preprocess_procs

    def prepare_data(self) -> None:
        max_seq_length = min(512, self.tokenizer.model_max_length)
        text_column_name = "text"
        # padding = "max_length" if data_args.pad_to_max_length else False
        padding = False

        if len(self.raw_datasets) == 1:
            complete_dataset = self.raw_datasets[0]["raw_dataset"]
            self.text_column_name = self.raw_datasets[0]["column_name"]
        else:
            # TODO
            self.text_column_name = self.raw_datasets[0]["column_name"]
            complete_dataset = concatenate_datasets([dataset["raw_dataset"] for dataset in self.raw_datasets], self.text_column_name)

        tokenizer = self.tokenizer

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line
                for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        column_names = list(complete_dataset["train"].features)

        tokenized_datasets = complete_dataset.map(
            tokenize_function,
            remove_columns=column_names,
            batched=True,
            num_proc=self.hparams.preprocess_procs,
            desc="Tokenizing",
        )

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (
                total_length // max_seq_length
            ) * max_seq_length  # Drop
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=self.hparams.preprocess_procs,
            desc="Grouping texts",
        )

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        self.data_train: Optional[Dataset] = tokenized_datasets["train"] if "train" in tokenized_datasets else None
        self.data_val: Optional[Dataset] = tokenized_datasets["validation"] if "validation" in tokenized_datasets else None
        self.data_test: Optional[Dataset] = tokenized_datasets["test"] if "test" in tokenized_datasets else None

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=HFDataset(self.data_train),
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=HFDataset(self.data_val),
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=HFDataset(self.data_test),
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
