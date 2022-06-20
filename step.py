from utils.decorators import Timer

import torch
import torch.nn.functional as F
from torchnet import meter

import logging


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
