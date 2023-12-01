import numpy as np


def print_DSCs(DSCs):
    return ' '.join([
        f'Class{cls+1}={DSC:.3f}' for cls, DSC in enumerate(DSCs)
    ] + [f'Mean={np.mean(DSCs):.3f}'])
    