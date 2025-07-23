from torch.utils.data import Dataset, DataLoader
from os.path import join
from .build import DATA_LOADER_REGISTRY
import numpy as np
import torch


class TimeSeriesFineTuningData(Dataset):

    def __init__(self, cfg, mode):
        self.mode = mode
        self.data = None
        self.fine_tuning_main_data = None
        self.fine_tuning_load_mode = cfg.DATA.FINE_TUNING_LOAD_MODE
        assert self.fine_tuning_load_mode in ["refresh", "full_period", "incremental"]

        if mode == "train":
            self.file_dir = cfg.DATA.TRAIN_DATA_DIR
            if cfg.DATA.FINE_TUNING_MAIN_DATA != "":
                self.fine_tuning_main_data = np.load(cfg.DATA.FINE_TUNING_MAIN_DATA)
                self.fine_tuning_main_data = torch.as_tensor(self.fine_tuning_main_data, dtype=torch.float32)


        if mode == "valid":
            self.file_dir = cfg.DATA.VALID_DATA_DIR
        if mode == "test":
            self.file_dir = cfg.DATA.TEST_DATA_DIR


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.data[idx, -1]

        sample = {
            "data": self.data[idx, :-1],
            "target": target,
        }
        return sample

    def update(self, date, file_type=".npy"):
        if self.fine_tuning_load_mode  == "refresh":
            self.refresh(date, file_type)
        elif self.fine_tuning_load_mode  == "full_period":
            new_data = np.load(join(self.file_dir, date + file_type))
            self.data = torch.cat([self.fine_tuning_main_data, new_data], dim=0)
        elif self.fine_tuning_load_mode  == "incremental":
            new_data = np.load(join(self.file_dir, date + file_type))
            self.data =  torch.cat([self.data, new_data], dim=0)
        else:
            raise ValueError(f"update_mode: {self.fine_tuning_load_mode } must be 'full_period' or 'incremental'")
        self.data = torch.as_tensor(self.data, dtype=torch.float32)
        return

    def refresh(self, date, file_type=".npy"):
        self.data = np.load(join(self.file_dir, date + file_type))
        self.data = torch.as_tensor(self.data, dtype=torch.float32)
        return



@DATA_LOADER_REGISTRY.register()
def phil_ts_dataloader(cfg, mode="train"):

    if mode == "train":
        dataset = TimeSeriesFineTuningData(cfg, mode)
    elif mode == "valid":
        dataset = TimeSeriesFineTuningData(cfg, mode)
    elif mode == "test":
        dataset = TimeSeriesFineTuningData(cfg, mode)
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