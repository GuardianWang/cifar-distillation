import os.path
import random

from model.get_model import get_model
from dataset.cifar100 import get_data
from optimizer import get_optimizer, get_scheduler
from utils.parser import get_cfg, cfg_to_str
from utils.meters import MinMaxMeter
from utils.decorators import Timer

from torch import nn
import torch
import torch.nn.functional as F
from torchnet import meter
from torchtoolbox.tools import summary

import ray
from ray import tune

import logging
from functools import partial
from copy import deepcopy
import psutil


@Timer
def train_step(model, criterion, optimizer, loader, device, cfg):

    ave_loss = meter.AverageValueMeter()
    model = model.train()
    for i, data in enumerate(loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        ave_loss.add(loss.item())
        if i % cfg.train_batch_print_freq == 0:
            logging.info(f"[train][batch {i:05d}][loss: {ave_loss.value()[0]:.4f}]")
    logging.info(f"[train][epoch loss: {ave_loss.value()[0]:.4f}]")
    return {
        "loss": ave_loss.value()[0],
    }


@Timer
def distill_step(teacher, student, hard_criterion, soft_criterion, optimizer, loader, device, cfg):

    ave_hard_loss = meter.AverageValueMeter()
    ave_soft_loss = meter.AverageValueMeter()
    ave_weighted_loss = meter.AverageValueMeter()
    student = student.train()
    for i, data in enumerate(loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        student_outputs = student(inputs)
        teacher_outputs = teacher(inputs)
        scale = 1 / cfg.temperature
        student_log_prob = F.log_softmax(student_outputs * scale, dim=-1)
        teacher_prob = F.softmax(teacher_outputs * scale, dim=-1)
        hard_loss = hard_criterion(student_outputs, labels)
        soft_loss = soft_criterion(student_log_prob, teacher_prob)
        loss = (1 - cfg.hard_weight) * soft_loss + cfg.hard_weight * hard_loss
        loss.backward()
        optimizer.step()
        ave_hard_loss.add(hard_loss.item())
        ave_soft_loss.add(soft_loss.item())
        ave_weighted_loss.add(loss.item())
        if i % cfg.train_batch_print_freq == 0:
            logging.info(f"[train][batch {i:05d}]"
                         f"[weighted loss: {ave_weighted_loss.value()[0]:.4f}]"
                         f"[hard loss: {ave_hard_loss.value()[0]:.4f}]"
                         f"[soft loss: {ave_soft_loss.value()[0]:.4f}]"
                         )
    logging.info(f"[train]"
                 f"[weighted loss: {ave_weighted_loss.value()[0]:.4f}]"
                 f"[hard loss: {ave_hard_loss.value()[0]:.4f}]"
                 f"[soft loss: {ave_soft_loss.value()[0]:.4f}]"
                 )
    return {
        "weighted_loss": ave_weighted_loss.value()[0],
        "hard_loss": ave_hard_loss.value()[0],
        "soft_loss": ave_soft_loss.value()[0],
    }


@Timer
def test_step(model, criterion, loader, device, cfg):
    ave_loss = meter.AverageValueMeter()
    accuracy = meter.ClassErrorMeter(topk=cfg.acc_top_k, accuracy=True)
    model = model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            ave_loss.add(loss.item())
            accuracy.add(outputs, labels)

    logging.info(f"[test][loss: {ave_loss.value()[0]:.4f}]")
    info = ["[test]"]
    for k, v in zip(cfg.acc_top_k, accuracy.value()):
        info.append(f"[top-{k} acc: {v:.4f}%]")
    logging.info("".join(info))
    return {
        "loss": ave_loss.value()[0],
        "acc": accuracy.value(),
    }


def train(cfg):
    if not torch.cuda.is_available() or cfg.not_use_gpu:
        device = torch.device("cpu")
    else:
        if cfg.benchmark:
            torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    logging.info(f"device: {device}")

    model = get_model(cfg, cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(optimizer, cfg)
    train_dataset, train_loader = get_data(root=cfg.data_root, train=True,
                                           batch_size=cfg.train_batch_size, extra_augment=cfg.extra_augment)
    test_dataset, test_loader = get_data(root=cfg.data_root, train=False, batch_size=cfg.test_batch_size)
    logging.info(f"model:\n{summary(model, torch.randn((1,) + test_dataset[0][0].shape, device=device))}")

    min_max_test_acc = MinMaxMeter()

    for epoch in range(cfg.train_epoch):
        logging.info(f"===epoch {epoch:04d}===")
        logging.info(f"lr: {scheduler.get_lr()}")
        train_loss = train_step(model, criterion, optimizer, train_loader, device=device, cfg=cfg)
        logging.info(f"[epoch train fps: {len(train_dataset) / train_step.last:.4f}]")
        if epoch % cfg.test_epoch_freq == 0:
            test_stat = test_step(model, criterion, test_loader, device=device, cfg=cfg)
            logging.info(f"[epoch test fps: {len(test_dataset) / test_step.last:.4f}]")
            is_test_best = epoch > 0 and test_stat["acc"][0] > min_max_test_acc.value(metric="max")
            min_max_test_acc.add(test_stat["acc"][0])

            if epoch > cfg.save_model_cooldown and is_test_best:
                torch.save(model.state_dict(), cfg.model_path)
        scheduler.step()

    logging.info(f"[ave train fps: {len(train_dataset) * train_step.ave_inv:.4f}]")
    logging.info(f"[ave test fps: {len(test_dataset) * test_step.ave_inv:.4f}]")


def tune_param(config: dict, cfg):
    cfg = deepcopy(cfg)
    # copy tune to cfg
    for k, v in config:
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


def distill(cfg):
    if not torch.cuda.is_available() or cfg.not_use_gpu:
        device = torch.device("cpu")
    else:
        if cfg.benchmark:
            torch.backends.cudnn.benchmark = True
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
    scheduler = get_scheduler(optimizer, cfg)
    train_dataset, train_loader = get_data(root=cfg.data_root, train=True,
                                           batch_size=cfg.train_batch_size, extra_augment=cfg.extra_augment)
    test_dataset, test_loader = get_data(root=cfg.data_root, train=False, batch_size=cfg.test_batch_size)
    logging.info(f"teacher:\n{summary(teacher, torch.randn((1,) + test_dataset[0][0].shape, device=device))}")
    logging.info(f"student:\n{summary(student, torch.randn((1,) + test_dataset[0][0].shape, device=device))}")
    logging.info(f"load teacher model weight {cfg.teacher_path}")
    teacher.load_state_dict(torch.load(cfg.teacher_path))

    min_max_test_acc = MinMaxMeter()

    for epoch in range(cfg.train_epoch):
        logging.info(f"===epoch {epoch:04d}===")
        logging.info(f"lr: {scheduler.get_lr()}")
        train_loss = distill_step(teacher, student, hard_criterion, soft_criterion,
                                  optimizer, train_loader, device, cfg)
        logging.info(f"[epoch train fps: {len(train_dataset) / distill_step.last:.4f}]")
        if epoch % cfg.test_epoch_freq == 0:
            test_stat = test_step(student, hard_criterion, test_loader, device=device, cfg=cfg)
            logging.info(f"[epoch test fps: {len(test_dataset) / test_step.last:.4f}]")
            is_test_best = epoch > 0 and test_stat["acc"][0] > min_max_test_acc.value(metric="max")
            min_max_test_acc.add(test_stat["acc"][0])
            if epoch > cfg.save_model_cooldown and is_test_best:
                torch.save(student.state_dict(), cfg.model_path)
        scheduler.step()

    logging.info(f"[ave train fps: {len(train_dataset) * distill_step.ave_inv:.4f}]")
    logging.info(f"[ave test fps: {len(test_dataset) * test_step.ave_inv:.4f}]")


def test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.not_use_gpu:
        device = torch.device("cpu")
    logging.info(f"device: {device}")
    model = get_model(cfg, cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    train_dataset, train_loader = get_data(root=cfg.data_root, train=True,
                                           batch_size=cfg.train_batch_size, augment_train=False)
    test_dataset, test_loader = get_data(root=cfg.data_root, train=False, batch_size=cfg.test_batch_size)
    logging.info(f"model:\n{summary(model, torch.randn((1,) + test_dataset[0][0].shape, device=device))}")
    logging.info(f"load model weight {cfg.model_path}")
    model.load_state_dict(torch.load(cfg.model_path))

    logging.info(f"===evaluate on test set===")
    test_step(model, criterion, test_loader, device=device, cfg=cfg)
    logging.info(f"[fps: {len(test_dataset) / test_step.last:.4f}]")

    logging.info(f"===evaluate on training set===")
    test_step(model, criterion, train_loader, device=device, cfg=cfg)
    logging.info(f"[fps: {len(train_dataset) / test_step.last:.4f}]")


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
    }
    config_str = {k: v.domain_str for k, v in config.items()}
    logging.info(f"tune params:\n{config_str}")

    scheduler = tune.schedulers.ASHAScheduler(
        metric="test_acc",
        mode="max",
        max_t=cfg.tune_num_epochs,
        grace_period=10,
        reduction_factor=2)
    reporter = tune.CLIReporter(
        metric_columns=["train_loss", "test_loss", "test_acc", "training_iteration"])
    result = tune.run(
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


def run():
    cfg = get_cfg()
    logging.basicConfig(filename=cfg.log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%b-%d-%y %H:%M:%S')
    logging.info(f"config:\n{cfg_to_str(cfg)}")
    if cfg.tune:
        run_tune(cfg)
    elif cfg.train:
        train(cfg)
        cfg.train_batch_size = cfg.test_batch_size
        test(cfg)
    elif cfg.distill:
        distill(cfg)
        cfg.train_batch_size = cfg.test_batch_size
        cfg.model = cfg.student
        test(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    run()
    pass
