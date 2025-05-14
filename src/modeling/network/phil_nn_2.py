import torch
import torch.nn as nn
from src.engine.build import device
from .build import NETWORK_REGISTRY

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class DeepResNetMLP(nn.Module):
    def __init__(self, input_dim, num_blocks=16, hidden_dim=22, output_dim=1, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


@NETWORK_REGISTRY.register()
def phil_nn_2(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    dropout = cfg.MODEL.DROPOUT if cfg else kwargs.get("DROPOUT")
    model = DeepResNetMLP(input_dim=input_dim, dropout=dropout)
    return model.to(device)

@NETWORK_REGISTRY.register()
def phil_rp_shallow_network_1(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    dropout = cfg.MODEL.DROPOUT if cfg else kwargs.get("DROPOUT")
    num_blocks = cfg.MODEL.NUM_BLOCKS if cfg else kwargs.get("NUM_BLOCKS")
    hidden_dim = cfg.MODEL.HIDDEN_DIM if cfg else kwargs.get("HIDDEN_DIM")
    model = DeepResNetMLP(input_dim=input_dim, num_blocks=num_blocks, hidden_dim=hidden_dim, dropout=dropout)
    return model.to(device)