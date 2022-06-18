from torchnet.meter.meter import Meter


class MinMaxMeter(Meter):
    def __init__(self):
        super().__init__()
        self.min = float('inf')
        self.max = -float('inf')

    def reset(self):
        self.min = float('inf')
        self.max = -float('inf')

    def add(self, value):
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def value(self, metric=""):
        if metric == "":
            return self.min, self.max
        elif metric == "min":
            return self.min
        elif metric == "max":
            return self.max
        else:
            raise ValueError("metric is either min or max or empty")
