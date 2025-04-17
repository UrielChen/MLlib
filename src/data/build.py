from src.config.registry import Registry

DATA_LOADER_REGISTRY = Registry("DATA_LOADER")


def build_dataloader(cfg):
    name = cfg.DATA_LOADER.NAME
    dataloader = DATA_LOADER_REGISTRY.get(name)

    if not cfg.TEST.TEST_ONLY and not cfg.TRAIN.TRAIN_ONLY:
        data_loader_dicts = {
            "train": dataloader(cfg, mode="train"),
            "valid": dataloader(cfg, mode="valid"),
            "test": dataloader(cfg, mode="test"),
        }
    elif cfg.TRAIN.TRAIN_ONLY:
        data_loader_dicts = {
            "train": dataloader(cfg, mode="train"),
            "valid": dataloader(cfg, mode="valid"),
        }
    elif cfg.TEST.TEST_ONLY:
        data_loader_dicts = {
            "test": dataloader(cfg, mode="test"),
        }
    else:
        raise RuntimeError("Unknown dataset")
    return data_loader_dicts
