import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display as ipd



from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

import sys

sys.path.append("/Users/vtokala/Documents/Research/di_nn")

from DCNN.datasets.base_dataset import BaseDataset
from DCNN.models.model_mse import DCNN
from DCNN.trainer import DCNNLightniningModule

GlobalHydra.instance().clear()
initialize(config_path="./di_nn/config")
config = compose("config")

MODEL_CHECKPOINT_PATH = '/Users/vtokala/Documents/Research/di_nn/DCNN/checkpoints/weights-epoch=19-validation_loss=-9.70.ckpt'
NOISY_DATASET_PATH = '/Users/vtokala/Documents/Research/di_nn/Dataset/noisy_testset_1f'
CLEAN_DATASET_PATH = '/Users/vtokala/Documents/Research/di_nn/Dataset/clean_testset_1f'



model = DCNNLightniningModule(config)
model.eval()
torch.set_grad_enabled(False)
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])

dataset = BaseDataset(NOISY_DATASET_PATH,CLEAN_DATASET_PATH)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=2
)

dataloader = iter(dataloader)

# while True:
#     try:
#         batch = next(dataloader)
#         #print(batch.shape)
#     except StopIteration:
#         break
#     model_output = model(batch[0])[0].numpy()
#     print(model_output.shape)
#     ipd.display(ipd.Audio(batch[0],rate=16000))
#     ipd.display(ipd.Audio(model_output,rate=16000))
#     #true_coords = batch[1]["source_coordinates"][0].numpy()

#librosa.output.write_wav('output.wav', model_output, 8000)