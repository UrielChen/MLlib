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

class TmpResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 64)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
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

@NETWORK_REGISTRY.register()
def phil_rp_model_e_wide_then_sparse(cfg=None, **kwargs):
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



def phil_rp_model_e_moderate_wide_then_sparse(cfg=None, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(856, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.blocks = nn.Sequential(
                TmpResidualBlock(),
                TmpResidualBlock(),
                TmpResidualBlock()
            )
            self.output = nn.Linear(64, 1)

        def forward(self, x):
            x = self.input(x)
            x = self.blocks(x)
            x = self.output(x)
            return x

    return Model().to(device)

def phil_rp_model_e_huge_wide_then_sparse(cfg=None, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(
                nn.Linear(856, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.blocks = nn.Sequential(
                TmpResidualBlock(),
                TmpResidualBlock(),
                TmpResidualBlock(),
                TmpResidualBlock(),
                TmpResidualBlock()
            )
            self.output = nn.Linear(128, 1)

        def forward(self, x):
            x = self.input(x)
            x = self.blocks(x)
            x = self.output(x)
            return x

    return Model().to(device)

