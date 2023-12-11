# Credits to Yin Cao et al:
# https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization/blob/master/models/model_utilities.py


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a convolutional or linear layer"""
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


def merge_list_of_dicts(list_of_dicts):
    result = {}

    def _add_to_dict(key, value):
        if len(value.shape) == 0: # 0-dimensional tensor
            value = value.unsqueeze(0)

        if key not in result:
            result[key] = value
        else:
            result[key] = torch.cat([
                result[key], value
            ])
    
    for d in list_of_dicts:
        for key, value in d.items():
            _add_to_dict(key, value)

    return result


def get_all_layers(model: nn.Module, layer_types=None, name_prefix=""):

    layers = {}
    
    for name, layer in model.named_children():
        if name_prefix:
            name = f"{name_prefix}.{name}"
        if isinstance(layer, nn.Sequential) or isinstance(layer, nn.ModuleList):
            layers.update(get_all_layers(layer, layer_types, name))
        else:
            layers[name] = layer
    
    if layer_types is not None:
        layers = {
            layer_id: layer
            for layer_id, layer in layers.items()
            if any([
                isinstance(layer, layer_type)
                for layer_type in layer_types
            ])
        }

    return layers
