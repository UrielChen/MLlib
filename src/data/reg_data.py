from torch.utils.data import Dataset, DataLoader
from .build import DATA_LOADER_REGISTRY
import numpy as np
import torch


class RegData(Dataset):

    def __init__(self, cfg, mode):
        if mode == "train":
            data = np.load(cfg.DATA.TRAIN_DATA)
            cfg.DATA.TRAIN_DATA_SHAPE = data.shape
        if mode == "valid":
            data = np.load(cfg.DATA.VALID_DATA)
            cfg.DATA.VALID_DATE_SHAPE = data.shape
        if mode == "test":
            data = np.load(cfg.DATA.TEST_DATA)
            cfg.DATA.TEST_DATA_SHAPE = data.shape
        self.data = data
        self.data = torch.as_tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.data[idx, -1]

        sample = {
            "data": self.data[idx, :-1],
            "target": target,
        }
        return sample


@DATA_LOADER_REGISTRY.register()
def phil_dataloader(cfg, mode="train"):

    if mode == "train":
        dataset = RegData(cfg, mode)
    elif mode == "valid":
        dataset = RegData(cfg, mode)
    elif mode == "test":
        dataset = RegData(cfg, mode)
    else:
        raise ValueError(
            "mode must be one of 'train' or 'valid' or test' "
        )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
    )
    return data_loader
