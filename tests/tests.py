from model.get_model import get_model
from utils.parser import get_cfg
from dataset.cifar100 import get_data, get_classes

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import torchvision
from torch import nn
import torch
import pytorch_warmup as warmup

import matplotlib.pyplot as plt
import numpy as np

import unittest


class Tests(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        self.cfg = get_cfg()

    def test_multistep_scheduler(self):
        cfg = self.cfg
        model = get_model(cfg=cfg, name=cfg.model)
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
        scheduler = MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
        lrs = []
        for i in range(1000):
            scheduler.step()
            lrs.append(scheduler.get_last_lr())
        plt.plot(range(1000), lrs)
        plt.show()
        plt.close('all')

    def test_cosineannealingwarmrestarts_scheduler(self):
        cfg = self.cfg
        model = get_model(cfg=cfg, name=cfg.model)
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        lrs = []
        for i in range(100):
            scheduler.step()
            lrs.append(scheduler.get_last_lr())
        plt.plot(range(100), lrs)
        plt.show()
        plt.close('all')

    def test_cosineannealingwarmrestarts_warmup_scheduler(self):
        cfg = self.cfg
        model = get_model(cfg=cfg, name=cfg.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: 1 / 10 * x)
        lrs = []
        for i in range(100):
            optimizer.step()
            if i < 10:
                warmup_scheduler.step()
            else:
                scheduler.step()
            lrs.append(get_lr(optimizer))
        plt.plot(range(100), lrs)
        plt.show()
        plt.close('all')

    def test_cosineannealingwarmrestarts_warmup_decay_scheduler(self):
        cfg = self.cfg
        model = get_model(cfg=cfg, name=cfg.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scheduler2 = lr_scheduler.MultiplicativeLR(optimizer, lambda x: 0.99 ** x)
        warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: 1 / 10 * x)
        lrs = []
        for i in range(100):
            optimizer.step()
            if i < 10:
                warmup_scheduler.step()
            else:
                scheduler.step()
                scheduler2.step()
            lrs.append(get_lr(optimizer))
        plt.plot(range(100), lrs)
        plt.show()
        plt.close('all')

    def test_cosineannealingwarmrestarts_exp_scheduler(self):
        cfg = self.cfg
        model = get_model(cfg=cfg, name=cfg.model)
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scheduler2 = lr_scheduler.MultiplicativeLR(optimizer, lambda x: 0.99 ** x)
        lrs = []
        for i in range(100):
            optimizer.step()
            scheduler.step()
            scheduler2.step()
            lrs.append(get_lr(optimizer))
        plt.plot(range(100), lrs)
        plt.show()
        plt.close('all')

    def test_cosineannealingwarmrestarts_step_scheduler(self):
        cfg = self.cfg
        model = get_model(cfg=cfg, name=cfg.model)
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scheduler2 = lr_scheduler.MultiplicativeLR(optimizer, lambda x: 0.8 ** (x // 10))
        lrs = []
        for i in range(100):
            optimizer.step()
            scheduler.step()
            scheduler2.step()
            lrs.append(get_lr(optimizer))
        plt.plot(range(100), lrs)
        plt.show()
        plt.close('all')

    def test_cosineannealinglr_scheduler(self):
        cfg = self.cfg
        model = get_model(cfg=cfg, name=cfg.model)
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        lrs = []
        for i in range(100):
            scheduler.step()
            lrs.append(scheduler.get_last_lr())
        plt.plot(range(100), lrs)
        plt.show()
        plt.close('all')

    def test_get_data(self, root=r"..", batch_size=4):
        def show_batch(loader):
            plt.close('all')
            # get some random training images
            dataiter = iter(loader)
            images, labels = dataiter.next()
            print(f"image shape: {images.shape}")
            print(f"labels shape: {labels.shape}")

            # print labels
            print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
            # show images
            imshow(torchvision.utils.make_grid(images))

        train_dataset, train_loader = get_data(root, train=True, batch_size=batch_size)
        test_dataset, test_loader = get_data(root, train=False, batch_size=batch_size)
        classes = get_classes(train_dataset)

        print("show train loader")
        show_batch(train_loader)
        print("show test loader")
        show_batch(test_loader)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]
