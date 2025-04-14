from src.config.registry import Registry


NETWORK_REGISTRY = Registry("NETWORK")
NETWORK_REGISTRY.__doc__ = """
Registry for networks, i.e. the whole model.
"""


def build_model(cfg):
    """ load network form function name

    """
    name = cfg.MODEL.NAME
    model = NETWORK_REGISTRY.get(name)(cfg)
    return model
