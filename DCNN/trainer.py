from omegaconf import OmegaConf
import torch

from torch.optim.lr_scheduler import MultiStepLR
from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
from DCNN.models.binaural_rnn_model import BinauralDCNN

from DCNN.models.model import DCNN
from DCNN.loss import BinauralLoss, Loss
from DCNN.utils.base_trainer import (
    BaseTrainer, BaseLightningModule
)


class DCNNTrainer(BaseTrainer):
    def __init__(self, config):
        lightning_module = DCNNLightningModule(config)
        super().__init__(lightning_module,
                         config["training"]["n_epochs"],
                         early_stopping_config=config["training"]["early_stopping"],
                         checkpoint_path=None,
                        #  strategy=config["training"]["strategy"],
                         accelerator=config["training"]["accelerator"])
                        # accelerator='mps')

    def fit(self, train_dataloaders, val_dataloaders=None):
        super().fit(self._lightning_module, train_dataloaders,
                    val_dataloaders=val_dataloaders)

    def test(self, test_dataloaders):
        super().test(self._lightning_module, test_dataloaders, ckpt_path="best")


class DCNNLightningModule(BaseLightningModule):
    """This class abstracts the
       training/validation/testing procedures
       used for training a DCNN
    """

    def __init__(self, config):
        config = OmegaConf.to_container(config)
        self.config = config

        if config["model"]["binaural"]:
            if config["model"]["attention"]:
                model = model = BinauralAttentionDCNN(**self.config["model"])
            else:
                model = BinauralDCNN(**self.config["model"])
            loss = BinauralLoss(
                rtf_weight=self.config["model"]["rtf_weight"],
                snr_weight=self.config["model"]["snr_weight"],
                ild_weight=self.config["model"]["ild_weight"],
                ipd_weight=self.config["model"]["ipd_weight"],
                avg_mode=config["model"]["binaural_avg_mode"],
                stoi_weight=config["model"]["stoi_weight"],
                kurt_weight=config["model"]["kurt_weight"],
                sdr_weight=self.config["model"]["sdr_weight"],
                mse_weight=self.config["model"]["mse_weight"],
                si_sdr_weight=self.config["model"]["si_sdr_weight"],
                si_snr_weight=self.config["model"]["si_snr_weight"],
                comp_loss_weight=self.config["model"]["comp_loss_weight"],
                msc_weight=self.config["model"]["msc_weight"]
                )
        else:    
            model = DCNN(**self.config["model"])

            loss = Loss(loss_mode=self.config["model"]["loss_mode"],
                        STOI_weight=self.config["model"]["STOI_weight"],
                        SNR_weight=self.config["model"]["snr_weight"])

        super().__init__(model, loss)

    def configure_optimizers(self):
        lr = self.config["training"]["learning_rate"]
        decay_step = self.config["training"]["learning_rate_decay_steps"]
        decay_value = self.config["training"]["learning_rate_decay_values"]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, decay_step, decay_value)

        return [optimizer], [scheduler]
