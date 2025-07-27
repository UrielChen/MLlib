import torch
import torch.nn as nn
from src.engine.build import device
from src.modeling.network.build import NETWORK_REGISTRY


# === Block 1: Plain Residual Block ===
class PlainResidualBlock(nn.Module):
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


# === Block 2: Bottleneck Residual Block ===
class BottleneckResidualBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim=64, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


# === Core Architecture ===
class DeepResNetMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=1,
        num_blocks=6,
        dropout=0.2,
        block_type="plain",           # "plain" or "bottleneck"
        bottleneck_dim=64             # used only for bottleneck block
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Block selector
        if block_type == "plain":
            Block = lambda: PlainResidualBlock(hidden_dim, dropout)
        elif block_type == "bottleneck":
            Block = lambda: BottleneckResidualBlock(hidden_dim, bottleneck_dim, dropout)
        else:
            raise ValueError(f"Unsupported block type: {block_type}")

        # Residual stack
        self.blocks = nn.Sequential(*[Block() for _ in range(num_blocks)])

        # Output head
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_layer(x)



@NETWORK_REGISTRY.register()
def new_model_a_bottleneck_deep_wide(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    return DeepResNetMLP(
        input_dim=input_dim,
        hidden_dim=256,
        num_blocks=6,
        dropout=0.2,
        block_type="bottleneck",
        bottleneck_dim=64
    ).to(device)


@NETWORK_REGISTRY.register()
def new_model_b_bottleneck_shallow(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    return DeepResNetMLP(input_dim=input_dim, hidden_dim=64, num_blocks=2, dropout=0.2,
                         block_type="bottleneck", bottleneck_dim=16).to(device)


@NETWORK_REGISTRY.register()
def new_model_c_bottleneck_deep_narrow(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    return DeepResNetMLP(input_dim=input_dim, hidden_dim=32, num_blocks=10, dropout=0.2,
                         block_type="plain").to(device)

