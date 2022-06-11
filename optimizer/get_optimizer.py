import torch.optim as optim


def get_optimizer(cfg, model):
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                      weight_decay=cfg.weight_decay)
    return opt
