from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps=4000, d_model=384, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch + 1
        scale = self.d_model ** (-0.5) * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
