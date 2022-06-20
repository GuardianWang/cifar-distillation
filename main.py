from model.get_model import get_model
from dataset.cifar100 import get_data
from optimizer import get_optimizer, get_scheduler
from utils.parser import get_cfg, cfg_to_str
from utils.meters import MinMaxMeter
from step import train_step, test_step, distill_step
from tune import run_tune

from torch import nn
import torch
from torchtoolbox.tools import summary

import logging


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
                                           batch_size=cfg.train_batch_size, extra_augment=cfg.extra_augment,
                                           cfg=cfg)
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


def run():
    cfg = get_cfg()
    logging.basicConfig(filename=cfg.log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%b-%d-%y %H:%M:%S')
    logging.info(f"config:\n{cfg_to_str(cfg)}")
    logging.info(f"see {torch.cuda.device_count()} gpus")
    if cfg.tune or cfg.tune_distill:
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
