import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchtoolbox.transform import Cutout

from multiprocessing import cpu_count


def get_data(root=r".", train=True, batch_size=4, augment_train=True, extra_augment=False,
             cfg=None):
    transform = []
    if train and augment_train:
        if extra_augment:
            if cfg:
                if cfg.ColorJitter:
                    transform.append(transforms.ColorJitter(
                        brightness=cfg.brightness,
                        contrast=cfg.contrast,
                        saturation=cfg.saturation,
                        hue=cfg.hue,
                    ))
                if cfg.RandomAffine:
                    transform.append(transforms.RandomAffine(
                        degrees=cfg.degrees,
                        translate=(0, cfg.translate_M),
                        scale=(cfg.scale_m, cfg.scale_M),
                        shear=cfg.shear,
                    ))
                if cfg.RandomPerspective:
                    transform.append(transforms.RandomPerspective(
                        distortion_scale=cfg.distortion_scale,
                        p=cfg.perspective_p,
                    ))
                if cfg.RandomGrayscale:
                    transform.append(transforms.RandomGrayscale(
                        p=cfg.gray_p
                    ))
            else:
                transform.extend([
                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                    transforms.RandomGrayscale(),
                ])
        transform.extend([
            transforms.RandomCrop(32, padding=4),
            Cutout(),
            transforms.RandomHorizontalFlip()
        ])
    transform.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform = transforms.Compose(transform)

    dataset = torchvision.datasets.CIFAR100(root=root, train=train,
                                            download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True if train else False,
                                             num_workers=cpu_count())

    return dataset, dataloader


def get_classes(dataset):
    return dataset.classes


if __name__ == "__main__":
    pass
