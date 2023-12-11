import torch

from .base_dataset import BaseDataset


def create_torch_dataloaders(config):
    return (
        create_torch_dataloader(config, "training"),
        create_torch_dataloader(config, "validation"),
        create_torch_dataloader(config, "test"),
    )


def create_torch_dataloader(config, mode):
    is_binaural = config["model"]["binaural"]

    if mode == "training":
        noisy_dataset_path = config["dataset"]["noisy_training_dataset_dir"]
        target_dataset_path = config["dataset"]["target_training_dataset_dir"]
        shuffle = True
    elif mode == "validation":
        noisy_dataset_path = config["dataset"]["noisy_validation_dataset_dir"]
        target_dataset_path = config["dataset"]["target_validation_dataset_dir"]
        shuffle = False
    elif mode == "test":
        noisy_dataset_path = config["dataset"]["noisy_test_dataset_dir"]
        target_dataset_path = config["dataset"]["target_test_dataset_dir"]
        shuffle = False
    elif mode == "test_rtf":
        noisy_dataset_path = config["test_rtf"]["noisy_dir"]
        target_dataset_path = config["test_rtf"]["target_dir"]
        shuffle = False
    
    if is_binaural or mode == "test_rtf":
        mono = False
    else:
        mono = True

    dataset = BaseDataset(noisy_dataset_path, target_dataset_path, mono=mono)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        pin_memory=config["training"]["pin_memory"],
        drop_last=False,
        num_workers=config["training"]["n_workers"]
    )
