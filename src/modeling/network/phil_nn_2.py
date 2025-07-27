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
    dropout = cfg.MODEL.DROPOUT if cfg else kwargs.get("DROPOUT")  # 0.2
    num_blocks = cfg.MODEL.NUM_BLOCKS if cfg else kwargs.get("NUM_BLOCKS")  # 16
    hidden_dim = cfg.MODEL.HIDDEN_DIM if cfg else kwargs.get("HIDDEN_DIM")  # 22
    model = DeepResNetMLP(input_dim=input_dim, num_blocks=num_blocks, hidden_dim=hidden_dim, dropout=dropout)
    return model.to(device)

@NETWORK_REGISTRY.register()
def phil_rp_shallow_network_2(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    dropout = 0.2
    num_blocks = 16
    hidden_dim = 32
    model = DeepResNetMLP(input_dim=input_dim, num_blocks=num_blocks, hidden_dim=hidden_dim, dropout=dropout)
    return model.to(device)

@NETWORK_REGISTRY.register()
def phil_rp_shallow_network_3(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    dropout = 0.2
    num_blocks = 20
    hidden_dim = 22
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
                nn.Linear(cfg.MODEL.INPUT_DIM, 24),
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
def phil_rp_model_f_wide_deep(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    return DeepResNetMLP(input_dim=input_dim, hidden_dim=128, num_blocks=6, dropout=0.2).to(device)


@NETWORK_REGISTRY.register()
def phil_rp_model_g_wide_shallow(cfg=None, **kwargs):
    input_dim = cfg.MODEL.INPUT_DIM if cfg else kwargs.get("INPUT_DIM")
    return DeepResNetMLP(input_dim=input_dim, hidden_dim=128, num_blocks=2, dropout=0.2).to(device)

@NETWORK_REGISTRY.register()
def phil_rp_model_h_deep_bottleneck(cfg=None, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cfg.MODEL.INPUT_DIM, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.net(x)
    return Model().to(device)


@NETWORK_REGISTRY.register()
def phil_rp_model_i_skip_projection(cfg=None, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(cfg.MODEL.INPUT_DIM, 128)
            self.bn = nn.BatchNorm1d(128)
            self.blocks = nn.Sequential(*[ResidualBlock(128, dropout=0.2) for _ in range(4)])
            self.output = nn.Linear(128 + 128, 1)  # concat input proj + residual output

        def forward(self, x):
            x_proj = self.bn(self.input_proj(x))
            h = self.blocks(x_proj)
            x_cat = torch.cat([x_proj, h], dim=1)
            return self.output(x_cat)
    return Model().to(device)



@NETWORK_REGISTRY.register()
def phil_rp_model_j_deep_narrow_but_gradual(cfg=None, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cfg.MODEL.INPUT_DIM, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )
        def forward(self, x):
            return self.net(x)
    return Model().to(device)


@NETWORK_REGISTRY.register()
def phil_rp_model_k_stack_residual_and_dense(cfg=None, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(cfg.MODEL.INPUT_DIM, 64)
            self.bn = nn.BatchNorm1d(64)
            self.blocks = nn.Sequential(
                ResidualBlock(64, dropout=0.2),
                ResidualBlock(64, dropout=0.2)
            )
            self.head = nn.Sequential(
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            x = self.bn(self.input(x))
            x = self.blocks(x)
            return self.head(x)
    return Model().to(device)



@NETWORK_REGISTRY.register()
def phil_rp_model_l_symmetric_autoencoder(cfg=None, **kwargs):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cfg.MODEL.INPUT_DIM, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        def forward(self, x):
            return self.net(x)
    return Model().to(device)


