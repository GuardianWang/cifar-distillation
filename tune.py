import random

from model.get_model import get_model
from dataset.cifar100 import get_data
from optimizer import get_optimizer, get_scheduler
from step import train_step, test_step, distill_step

from torch import nn
import torch
from torchtoolbox.tools import summary

from ray import tune

import logging
from functools import partial
from copy import deepcopy
import psutil


def tune_param(config: dict, cfg):
    cfg = deepcopy(cfg)
    # copy tune to cfg
    for k, v in config.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise ValueError(f"{k} not recognized")

    if not torch.cuda.is_available() or cfg.not_use_gpu:
        device = torch.device("cpu")
    else:
        if cfg.benchmark:
            # https://github.com/ray-project/ray/issues/8569#issuecomment-1139534091
            eval('setattr(torch.backends.cudnn, "benchmark", True)')
        device = torch.device("cuda")
    logging.info(f"device: {device}")

    model = get_model(cfg, cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(cfg, model)
    train_dataset, train_loader = get_data(root=cfg.data_root, train=True,
                                           batch_size=cfg.train_batch_size, extra_augment=cfg.extra_augment,
                                           cfg=cfg)
    test_dataset, test_loader = get_data(root=cfg.data_root, train=False, batch_size=cfg.test_batch_size)
    logging.info(f"model:\n{summary(model, torch.randn((1,) + test_dataset[0][0].shape, device=device))}")

    for epoch in range(cfg.tune_num_epochs):
        logging.info(f"===epoch {epoch:04d}===")
        logging.info(f"lr: {cfg.lr}")
        train_loss = train_step(model, criterion, optimizer, train_loader, device=device, cfg=cfg)
        test_loss = test_step(model, criterion, test_loader, device=device, cfg=cfg)
        tune.report(train_loss=train_loss["loss"], test_loss=test_loss["loss"], test_acc=test_loss["acc"][0])


def tune_distill_param(config: dict, cfg):
    cfg = deepcopy(cfg)
    # copy tune to cfg
    for k, v in config.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            raise ValueError(f"{k} not recognized")

    if not torch.cuda.is_available() or cfg.not_use_gpu:
        device = torch.device("cpu")
    else:
        if cfg.benchmark:
            eval('setattr(torch.backends.cudnn, "benchmark", True)')
        device = torch.device("cuda")
    logging.info(f"device: {device}")

    teacher = get_model(cfg, cfg.teacher).to(device).eval()
    # freeze teacher model
    for param in teacher.parameters():
        param.requires_grad = False
    student = get_model(cfg, cfg.student).to(device)

    hard_criterion = nn.CrossEntropyLoss()
    soft_criterion = nn.KLDivLoss()
    optimizer = get_optimizer(cfg, student)

    train_dataset, train_loader = get_data(root=cfg.data_root, train=True,
                                           batch_size=cfg.train_batch_size, extra_augment=cfg.extra_augment,
                                           cfg=cfg)
    test_dataset, test_loader = get_data(root=cfg.data_root, train=False, batch_size=cfg.test_batch_size)

    for epoch in range(cfg.tune_num_epochs):
        train_loss = distill_step(teacher, student, hard_criterion, soft_criterion,
                                  optimizer, train_loader, device, cfg)
        test_loss = test_step(student, hard_criterion, test_loader, device=device, cfg=cfg)
        tune.report(train_loss=train_loss["weighted_loss"], test_loss=test_loss["loss"], test_acc=test_loss["acc"][0])


def run_tune(cfg):
    config = {
        # optimizer
        "optimizer": tune.choice(["adamw", "adam"]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "amsgrad": tune.choice([True, False]),
        "beta1": tune.sample_from(lambda _: 1 - pow(10, random.uniform(-3, -0.3))),
        "beta2": tune.sample_from(lambda _: 1 - pow(10, random.uniform(-3, -1))),
        "train_batch_size": tune.choice([32, 64, 128, 256]),
    }
    if cfg.extra_augment:
        config.update({
            # data augmentation
            "ColorJitter": tune.choice([True, False]),
            "brightness": tune.uniform(0, 1),
            "contrast": tune.uniform(0, 1),
            "saturation": tune.uniform(0, 1),
            "hue": tune.uniform(0, 0.5),

            "RandomAffine": tune.choice([True, False]),
            "degrees": tune.uniform(0, 180),
            "translate_M": tune.uniform(0, 1),
            "scale_m": tune.uniform(0.5, 1),
            "scale_M": tune.uniform(1, 2),
            "shear": tune.uniform(0, 90),

            "RandomPerspective": tune.choice([True, False]),
            "distortion_scale": tune.uniform(0, 1),
            "perspective_p": tune.uniform(0, 1),

            "RandomGrayscale": tune.choice([True, False]),
            "gray_p": tune.uniform(0, 1),
        })
    if cfg.distill:
        config.update({
            "hard_weight": tune.uniform(0, 0.5),
            "temperature": tune.uniform(1, 50)
        })
    config_str = {k: v.domain_str for k, v in config.items()}
    logging.info(f"tune params:\n{config_str}")

    scheduler = tune.schedulers.ASHAScheduler(
        metric="test_acc",
        mode="max",
        max_t=cfg.tune_num_epochs,
        grace_period=cfg.tune_grace_period,
        reduction_factor=2)
    reporter = tune.CLIReporter(
        metric_columns=["train_loss", "test_loss", "test_acc", "training_iteration"])
    result = tune.run(
        partial(tune_distill_param, cfg=cfg) if cfg.tune_distill else
        partial(tune_param, cfg=cfg),
        resources_per_trial={
            "cpu": psutil.cpu_count(True) // torch.cuda.device_count(),
            "gpu": 0 if cfg.not_use_gpu else 1,
        },
        config=config,
        num_samples=cfg.tune_num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        log_to_file=True,
        local_dir=cfg.ray_local_dir,
    )

    best_trial = result.get_best_trial("test_acc", "max", "all")
    logging.info("Best trial config: {}".format(best_trial.config))
    logging.info("Best trial final test acc: {}".format(
        best_trial.last_result["test_acc"]))
    logging.info("Best trial final test loss: {}".format(
        best_trial.last_result["test_loss"]))
    logging.info("Best trial final train loss: {}".format(
        best_trial.last_result["train_loss"]))
