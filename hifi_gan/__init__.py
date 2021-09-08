from .model import Generator as HiFiGANModel
from .utils import get_hparams_from_dir, latest_checkpoint_path, load_checkpoint


def load_hifi_gan(model_dir):
    hps = get_hparams_from_dir(model_dir)
    checkpoint_path = latest_checkpoint_path(model_dir)
    model = HiFiGANModel(hps)
    load_checkpoint(checkpoint_path, model)
    return model
