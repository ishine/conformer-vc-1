import glob
import json
import os

import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_hparams_from_dir(model_dir):
    config_file = os.path.join(model_dir, 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h


def latest_checkpoint_path(model_dir, regex='g_*'):
    f_list = glob.glob(os.path.join(model_dir, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def load_checkpoint(checkpoint_path, model):
    state_dict_g = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict_g['generator'])
    model.eval()
    model.remove_weight_norm()
    return model


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
