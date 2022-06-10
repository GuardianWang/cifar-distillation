from model import *


def get_model(cfg, name):
    if name in ["resnet18", "resnet50", "resnet101", "resnet152"]:
        return resnet.ResNet(name=name, cfg=cfg)
    elif name in ["resnet_cifar"]:
        return resnet_original.ResNetOriginal(cfg)
    else:
        raise ValueError(f"{name} not defined")
