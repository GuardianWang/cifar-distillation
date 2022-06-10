from utils.parser import get_cfg

import torchvision.models as models
from torch import nn, randn
import torch

from torchtoolbox.tools import summary


class ResNet(nn.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.cfg = cfg
        if name == "resnet18":
            resnet = models.resnet18(pretrained=False)
        elif name == "resnet50":
            resnet = models.resnet50(pretrained=False)
        elif name == "resnet101":
            resnet = models.resnet101(pretrained=False)
        elif name == "resnet152":
            resnet = models.resnet152(pretrained=False)
        else:
            raise ValueError(f"{name} not defined")
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, cfg.classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_resnet(in_shape=(1, 3, 32, 32)):
    cfg = get_cfg()
    model = ResNet(cfg)
    data = randn(in_shape)
    print(summary(model, data))


if __name__ == "__main__":
    test_resnet()
