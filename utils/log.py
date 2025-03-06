import logging
from collections import OrderedDict
import time
import torch


def get_logger(file_name: str, mode="a"):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    # remove old handlers to avoid duplicated outputs.
    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(logging.FileHandler(file_name, mode=mode))
    logger.addHandler(logging.StreamHandler())
    return logger


class Averager(object):
    def __init__(self, *keys):
        """An averager to record numeric metrics.

        Attrs:
            sum, OrderedDict[key, sum]: sum for each metric.
            cnt,  OrderedDict[key, cnt]: number of records for each metric.
            clock, float: record time (secs).
        """
        self.sum = OrderedDict()
        self.cnt = OrderedDict()
        self.clock = time.time()
        for key in keys:
            self.sum[key] = 0
            self.cnt[key] = 0

    def update(self, key, val):
        if isinstance(val, torch.Tensor):
            val = val.item()

        if self.sum.get(key, None) is None:
            self.sum[key] = val
            self.cnt[key] = 1
        else:
            self.sum[key] = self.sum[key] + val
            self.cnt[key] += 1

    def reset(self):
        for key in self.sum.keys():
            self.sum[key] = 0
            self.cnt[key] = 0

        self.clock = time.time()

    def info(self):
        """Write information line for current records."""
        line = ""
        for key in self.sum.keys():
            val = self.sum[key] / self.cnt[key]  # average
            line += f"{key}: {val:.4f} "

        line += f"({time.time()-self.clock:.3f} secs)"
        return line
