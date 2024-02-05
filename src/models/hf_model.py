from typing import Tuple

from lightning import LightningModule

import torch
from transformers import AutoModelForCausalLM
from torchmetrics import MeanMetric


class HFModelForCausalLM(LightningModule):

    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.model_type = "hf"
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        self.train_loss = MeanMetric()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
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
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)


