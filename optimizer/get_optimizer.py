import torch.optim as optim


def get_optimizer(cfg, model):
    name = cfg.optimizer
    if name == "adamw":
        opt = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                          weight_decay=cfg.weight_decay, amsgrad=cfg.amsgrad)
    elif name == "adam":
        opt = optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                         weight_decay=cfg.weight_decay)
    elif name == "sgd":
        opt = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                        weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    else:
        raise ValueError(f"optimizer {name} not defined")
    return opt
