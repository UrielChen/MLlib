import torch
import torch.nn as nn
from src.engine.build import device
from .build import NETWORK_REGISTRY


class AlexNet1D(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 11), stride=4, padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=2),
            nn.Conv2d(64, 192, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=2),
            nn.Conv2d(192, 384, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@NETWORK_REGISTRY.register()
def alexnet_1d(cfg=None, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet1D(num_classes=3, **kwargs)
    return model.to(device)

