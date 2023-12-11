import torch
import torchaudio
import os

from pathlib import Path


SR = 16000
# N_MICROPHONE_SECONDS = 1
# N_MICS = 4


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 noisy_dataset_dir,
                 target_dataset_dir,
                 sr=SR,
                 mono=False):

        self.sr = sr
        self.target_dataset_dir = target_dataset_dir
        self.noisy_dataset_dir = noisy_dataset_dir

        self.mono = mono

        self.noisy_file_paths = self._get_file_paths(noisy_dataset_dir)
        self.target_file_paths = self._get_file_paths(target_dataset_dir)
      
    def __len__(self):
        return len(self.noisy_file_paths)

    def __getitem__(self, index):
        clean_audio_sample_path = self.target_file_paths[index]
        noisy_audio_sample_path = self.noisy_file_paths[index]
        #path = os.path.dirname(self.audio_dir)
        clean_signal, _ = torchaudio.load(clean_audio_sample_path)
        # breakpoint()

        noisy_signal, _ = torchaudio.load(noisy_audio_sample_path)
        

        if self.mono:
            return (noisy_signal[0], clean_signal[0])
        else:
            return (noisy_signal, clean_signal, clean_audio_sample_path, noisy_audio_sample_path)

    def _get_file_paths(self, dataset_dir):
        file_paths = [
            os.path.join(dataset_dir / fp) for fp in sorted(Path(dataset_dir).rglob('*.wav'))
        ]
        return file_paths
