import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchtoolbox.transform import Cutout

from multiprocessing import cpu_count


def get_data(root=r".", train=True, batch_size=4, augment_train=True):
    transform = []
    if train and augment_train:
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
