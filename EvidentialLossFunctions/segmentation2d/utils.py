from config import cfg


def print_DSCs(DSCs):
    DSCs = DSCs.flatten()
    assert DSCs.shape[0] == len(cfg.DATASET.CLASS_NAMES)
    return ' '.join([
        f'{class_name}={DSC.item():.3f}' for class_name, DSC in zip(cfg.DATASET.CLASS_NAMES, DSCs)
    ] + [f'Mean={DSCs.mean().item():.3f}'])
    