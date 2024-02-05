from typing import Tuple, Dict

from lightning import LightningModule

import torch
from transformers import AutoModelForCausalLM
from torchmetrics import MeanMetric
from torchmetrics.text import Perplexity


class HFModelForCausalLM(LightningModule):

    def __init__(self, model_name_or_path: str, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model_type = "hf"
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        self.train_loss = MeanMetric()
        # TODO
        # self.train_ppl = Perplexity()

    def model_step(self, batch: Dict[str, torch.Tensor]):
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        # Shift tokens to the right is handled by the model, no need to do it manually
        output = self.model.forward(input_ids, attention_mask=attention_mask, labels=input_ids)
        return output.loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
