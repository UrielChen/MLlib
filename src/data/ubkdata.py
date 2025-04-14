import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from .build import DATA_LOADER_REGISTRY


class UBKData(Dataset):

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
        # self.data = data[:2000]
        self.data = data
        self.data = torch.as_tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx, -1].clip(min=0).type(torch.LongTensor)

        sample = {
            "data": self.data[idx, :-1],
            "label": label,
        }
        return sample

    def get_labels(self):
        return self.data[:, -1]


@DATA_LOADER_REGISTRY.register()
def ubk_dataloader(cfg, mode="train"):
    from torchsampler import ImbalancedDatasetSampler

    sampler = None
    if mode == "train":
        dataset = UBKData(cfg, mode)
        sampler = ImbalancedDatasetSampler(dataset)
    elif mode == "valid":
        dataset = UBKData(cfg, mode)
    elif mode == "test":
        dataset = UBKData(cfg, mode)
    else:
        raise ValueError(
            "mode must be one of 'train' or 'valid' or test' "
        )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg.DATA_LOADER.TRAIN_BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        sampler=sampler,
    )
    return data_loader