import torch
import torch.nn as nn
from .build import NETWORK_REGISTRY


class MLP(nn.Module):
    def __init__(self,input_dim=194, output_dim=3):
        super().__init__()
        self.inp_layer = nn.Linear(input_dim, 256)
        self.relu_1 = nn.LeakyReLU()

        self.layer_2 = nn.Linear(256, 512)
        self.relu_2 = nn.LeakyReLU()

        self.layer_3 = nn.Linear(512, 512)
        self.relu_3 = nn.LeakyReLU()

        self.drop_out = nn.Dropout(0.2)
        self.out_layer = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.relu_1(self.inp_layer(x))
        x = self.relu_2(self.layer_2(x))
        x = self.relu_3(self.layer_3(x))
        x = self.drop_out(x)
        x = self.out_layer(x)
        return x


@NETWORK_REGISTRY.register()
def mlp_model(cfg=None):
    model = MLP()
    model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model
