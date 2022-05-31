from model import *


def get_model(cfg):
    if cfg.model in ["resnet18", "resnet50", "resnet101", "resnet152"]:
        return resnet.ResNet(cfg)
    elif cfg.model in ["resnet_cifar"]:
        return resnet_original.ResNetOriginal(cfg)
    else:
        raise ValueError(f"{cfg.model} not defined")
