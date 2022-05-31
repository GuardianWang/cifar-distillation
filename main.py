from model.resnet import ResNet
from model.resnet_original import ResNetOriginal
from dataset.cifar100 import get_data
from utils.parser import get_cfg, cfg_to_str

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch import nn
import torch
from torchnet import meter

import logging


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
            logging.info(f"[train][batch {i:05d}][batch loss: {ave_loss.value()[0]:.4f}]")
    logging.info(f"[train][epoch loss: {ave_loss.value()[0]:.4f}]")
    return ave_loss.value()[0]


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
    return ave_loss.value()[0]


def train(cfg):
    if not torch.cuda.is_available() or cfg.not_use_gpu:
        device = torch.device("cpu")
    else:
        if cfg.benchmark:
            torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    logging.info(f"device: {device}")

    model = ResNetOriginal(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    scheduler = MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
    train_dataset, train_loader = get_data(root=cfg.data_root, train=True, batch_size=cfg.train_batch_size)
    test_dataset, test_loader = get_data(root=cfg.data_root, train=False, batch_size=cfg.test_batch_size)

    for epoch in range(cfg.train_epoch):
        logging.info(f"===epoch {epoch:04d}===")
        logging.info(f"lr: {scheduler.get_lr()}")
        train_loss = train_step(model, criterion, optimizer, train_loader, device=device, cfg=cfg)
        if epoch % cfg.test_epoch_freq == 0:
            test_step(model, criterion, test_loader, device=device, cfg=cfg)
        if epoch > 0 and epoch % cfg.save_model_freq == 0:
            torch.save(model.state_dict(), cfg.model_path)
        scheduler.step(epoch)


def test(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg.not_use_gpu:
        device = torch.device("cpu")
    logging.info(f"device: {device}")
    model = ResNetOriginal(cfg).to(device)
    logging.info(f"load model weight {cfg.model_path}")
    model.load_state_dict(torch.load(cfg.model_path))
    criterion = nn.CrossEntropyLoss()
    train_dataset, train_loader = get_data(root=cfg.data_root, train=True,
                                           batch_size=cfg.train_batch_size, augment_train=False)
    test_dataset, test_loader = get_data(root=cfg.data_root, train=False, batch_size=cfg.test_batch_size)
    logging.info(f"===evaluate on test set===")
    test_step(model, criterion, test_loader, device=device, cfg=cfg)
    logging.info(f"===evaluate on training set===")
    test_step(model, criterion, train_loader, device=device, cfg=cfg)


def run():
    cfg = get_cfg()
    logging.basicConfig(filename=cfg.log_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s', datefmt='%b-%d-%y %H:%M:%S')
    logging.info(f"config:\n{cfg_to_str(cfg)}")
    if cfg.train:
        train(cfg)
    else:
        test(cfg)


if __name__ == "__main__":
    run()
    pass
