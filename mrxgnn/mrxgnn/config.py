import copy
import yaml


def load(path, overrides=()):
    cfg = yaml.safe_load(open(path))
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg, overrides):
    """overrides like ["model.use_pet=false", "out_dir=results/x", "cv.seeds=[0]"]"""
    cfg = copy.deepcopy(cfg)
    for o in overrides:
        k, v = o.split("=", 1)
        node = cfg
        *path, last = k.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(v)
    return cfg
