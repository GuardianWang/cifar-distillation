from torch.optim import lr_scheduler


class Scheduler:
    def __init__(self, warmup=None, warmup_iter=0, body=None, multiplier=None):
        self.warmup = warmup
        self.warmup_iter = warmup_iter
        self.body = body
        self.multiplier = multiplier
        self.epoch = 0

    def step(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch

        if self.warmup_iter > 0 and self.epoch < self.warmup_iter:
            self.warmup.step()
        else:
            self.body.step()
            self.multiplier.step()

        self.epoch += 1

    def get_lr(self):
        return get_lr(self.body.optimizer)


def get_scheduler(optimizer, cfg):
    body = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult)
    multiplier = lr_scheduler.MultiplicativeLR(optimizer, lambda x: cfg.mult_gamma ** x)
    warmup = lr_scheduler.LambdaLR(optimizer, lambda x: 1 / (cfg.warmup_iter + 1) * (x + 1))
    scheduler = Scheduler(warmup, cfg.warmup_iter, body, multiplier)

    return scheduler


def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]
