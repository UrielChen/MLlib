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

@NETWORK_REGISTRY.register()
def phil_rp_model_a_shallow_wide(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    return DeepResNetMLP(input_dim=input_dim, hidden_dim=28, num_blocks=1, dropout=0.2).to(device)

@NETWORK_REGISTRY.register()
def phil_rp_model_b_moderate_depth(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    return DeepResNetMLP(input_dim=input_dim, hidden_dim=16, num_blocks=3, dropout=0.2).to(device)

@NETWORK_REGISTRY.register()
def phil_rp_model_c_deep_narrow(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    return DeepResNetMLP(input_dim=input_dim, hidden_dim=8, num_blocks=6, dropout=0.2).to(device)

@NETWORK_REGISTRY.register()
def phil_rp_model_d_encoder_decoder(cfg=None, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(856, 24),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(24, 6),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(6, 24),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(24, 1)
            )
        def forward(self, x):
            return self.net(x)
    return Model().to(device)

@NETWORK_REGISTRY.register()
def phil_rp_model_e_wide_then_sparse(cfg=None, **kwargs):
    class TmpResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(20, 5)
            self.linear2 = nn.Linear(5, 20)
            self.bn1 = nn.BatchNorm1d(5)
            self.bn2 = nn.BatchNorm1d(20)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            identity = x
            out = self.linear1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.linear2(out)
            out = self.bn2(out)
            out = self.dropout(out)
            return self.relu(out + identity)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(856, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.blocks = nn.Sequential(
                TmpResidualBlock(),
                TmpResidualBlock()
            )
            self.output = nn.Linear(20, 1)

        def forward(self, x):
            x = self.input(x)
            x = self.blocks(x)
            x = self.output(x)
            return x

    return Model().to(device)



