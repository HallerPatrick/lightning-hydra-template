
from lightning.pytorch.callbacks import Callback

from src.utils import RankedLogger


class HFCheckpointCallback(Callback):

    def on_fit_end(self, trainer, pl_module):
        """Called when the fit ends."""
        if hasattr(trainer.model, "model_type") and trainer.model.model_type == "hf":
            trainer.model.model.save_pretrained(trainer.checkpoint_callback.dirpath)
            RankedLogger(__name__, rank_zero_only=True).info(f"Saved HF model to {trainer.checkpoint_callback.dirpath}")
