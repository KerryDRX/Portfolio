from torch import nn
from edl.model import DenseNormalGamma


def build_model():
    return nn.Sequential(
        nn.Linear(1, 100),
        nn.ReLU(True),
        nn.Linear(100, 100),
        nn.ReLU(True),
        DenseNormalGamma(100, 1),
    )
