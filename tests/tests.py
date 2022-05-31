from model.get_model import get_model
from utils.parser import get_cfg

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch import nn
import torch

import matplotlib.pyplot as plt

import unittest


class Tests(unittest.TestCase):
    def __init__(self, method_name='runTest'):
        super().__init__(method_name)
        self.cfg = get_cfg()

    def test_scheduler(self):
        cfg = self.cfg
        model = get_model(cfg)
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
