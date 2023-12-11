from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


def load_config(key=None, reset_hydra=True):
    if reset_hydra:
        GlobalHydra.instance().clear()
    
    initialize(config_path="../../config", job_name="_")
    cfg = compose(config_name="config")

    if key is not None:
        return cfg[key]

    return cfg
