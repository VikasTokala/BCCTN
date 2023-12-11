import pickle
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import (TQDMProgressBar,
                                         ModelCheckpoint, EarlyStopping
                                         )
from pytorch_lightning import loggers as pl_loggers

from DCNN.utils.model_utilities import merge_list_of_dicts



SAVE_DIR = "logs/"


class BaseTrainer(pl.Trainer):
    def __init__(self, lightning_module, n_epochs,
                 use_checkpoint_callback=True, checkpoint_path=None,
                 early_stopping_config=None,strategy="ddp_notebook",
                 accelerator='None', profiler='advanced'):

        gpu_count = torch.cuda.device_count()

        if accelerator is None:
            accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        # if accelerator == "mac":
        #     accelerator = "auto" 


        strategy = strategy if gpu_count > 1 else None

        progress_bar = CustomProgressBar()
        early_stopping = EarlyStopping(early_stopping_config["key_to_monitor"],
                                       early_stopping_config["min_delta"],
                                       early_stopping_config["patience_in_epochs"]
                                       )
        checkpoint_callback = ModelCheckpoint(
            monitor="validation_loss",
            save_last=True,
            save_weights_only=True
        )
        # pl.metrics.Accuracy(compute_on_step=False)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=SAVE_DIR)
        # csv_logger = pl_loggers.CSVLogger(save_dir=SAVE_DIR)

        callbacks = [early_stopping]  # feature_map_callback],
        if use_checkpoint_callback:
            callbacks.append(checkpoint_callback)

        super().__init__(
            max_epochs=n_epochs,
            callbacks=[progress_bar,
                       checkpoint_callback  # feature_map_callback
                       ],
            logger=[tb_logger], # csv_logger],
            accelerator=accelerator,
            strategy=strategy,
            gpus=gpu_count,
            log_every_n_steps=400, enable_progress_bar=True, detect_anomaly=False)

        if checkpoint_path is not None:
            _load_checkpoint(lightning_module.model, checkpoint_path)


        self._lightning_module = lightning_module


class BaseLightningModule(pl.LightningModule):
    """Class which abstracts interactions with Hydra
    and basic training/testing/validation conventions
    """

    def __init__(self, model, loss,
                 log_step=400):
        super().__init__()

        self.is_cuda_available = torch.cuda.is_available()

        self.model = model
        self.loss = loss

        self.log_step = log_step

    def _step(self, batch, batch_idx, log_model_output=False,
              log_labels=False):

        x, y = batch
        # 1. Compute model output and loss
        output = self.model(x)
        # model_target = self.model(y)
        loss = self.loss(output, y)
        # from GPUtil import showUtilization as gpu_usage

        output_dict = {
            "loss": loss
        }

        # TODO: Add these to a callback
        # 2. Log model output
        if log_model_output:

            output_dict["model_output"] = output

        # 3. Log step metrics
        self.log("loss_epoch", output_dict["loss"],
                 on_step=False, prog_bar=False,on_epoch=True)

        return output_dict

    def training_step(self, batch, batch_idx):

        return self._step(batch, batch_idx,log_model_output=False)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx,
                          log_model_output=False, log_labels=True)

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx,
                          log_model_output=False, log_labels=True)

    def _epoch_end(self, outputs, epoch_type="train", save_pickle=False):
        # 1. Compute epoch metrics
        outputs = merge_list_of_dicts(outputs)
        epoch_stats = {
            f"{epoch_type}_loss": outputs["loss"].mean(),
            f"{epoch_type}_std": outputs["loss"].std()
            
        }

        # outputs.detach()
        # 2. Log epoch metrics
        for key, value in epoch_stats.items():

            self.log(key, value, on_epoch=True, prog_bar=True)

        # 3. Save complete epoch data on pickle
        if save_pickle:
            pickle_filename = f"{epoch_type}.pickle"
            with open(pickle_filename, "wb") as f:
                pickle.dump(outputs, f)

        return epoch_stats

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, epoch_type="validation")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, epoch_type="test", save_pickle=True)

    def forward(self, x):
        return self.model(x)

    def fit(self, dataset_train, dataset_val):
        super().fit(self.model, dataset_train, val_dataloaders=dataset_val)

    def test(self, dataset_test, ckpt_path="best"):
        super().test(self.model, dataset_test, ckpt_path=ckpt_path)


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def _load_checkpoint(model, checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = {}

    for k, v in checkpoint["state_dict"].items():
        k = k.replace("model.", "")
        state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)
