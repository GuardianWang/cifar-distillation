from time import time

from torchnet import meter


class Timer:
    def __init__(self, func):
        self.func = func
        self._ave = meter.AverageValueMeter()
        self._ave_inv = meter.AverageValueMeter()
        self._last = 0

    def __call__(self, *args, **kwargs):
        start_t = time()
        res = self.func(*args, **kwargs)
        duration = time() - start_t
        self._ave.add(duration)
        self._ave_inv.add(1 / (duration + 1e-8))
        self._last = duration

        return res

    def reset(self):
        self._ave.reset()
        self._last = 0

    @property
    def ave(self):
        return self._ave.value()[0]

    @property
    def ave_inv(self):
        return self._ave_inv.value()[0]

    @property
    def last(self):
        return self._last
