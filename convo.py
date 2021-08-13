"""
convo.py

A package to automatically set up simple convolutional neural networks in pytorch.

Thomas Findlay, 08/2021

findlaytel@gmail.com
"""

import torch.nn as nn
from collections import OrderedDict
from math import log

class _LastUpdatedOrderedDict(OrderedDict):
    """Store items in the order the keys were last added"""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        

def make_convolutions(in_shape, out_shape, n_layers, kernel_size=None, stride=1,
                        padding=0, padding_mode="zeros", dilation=0, bias=True, 
                        activation="relu", pool_type="max", norm_type=None):
    """
    kwargs
    ========
    pool_type : "max" or "avg", default "max"
    norm_type : None, "batch"
    kernel_size : int or iter length n_layers
    stride : ""
    padding : ""
    padding_mode : 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
    dilation : int or iter length n_layers
    bias : bool
    activation: fcn
    """
    
    _check_shapes(in_shape, out_shape)
    
    kernel_size = _parse_kernel_size(kernel_size) ## TBD
    
    layers = _LastUpdatedOrderedDict()
   
    n_dims = len(in_shape) - 1 # SPATIAL dims
    
    dim_shapes = np.zeros((n_layers, n_dims))
    for dim in range(n_dims + 1):
        shapes[:, dim] = np.array(_get_dim_sizes(in_shape[dim], out_shape[dim], n_layers))
    
    conv_layer_type = _get_conv_layer_type(n_dims)
    pool_layer_type = _get_pool_layer_type(pool_type, n_dims)
    norm_layer_type = _get_norm_layer_type(norm_type, n_dims)
    
    for n in range(n_layers):
        layers[f"conv_{n + 1}"] = conv_layer_type()
        layers[f"pool_{n + 1}"] = pool_layer_type()
        if norm_type is not None:
            layers[f"{norm_layer_type}_norm_{n + 1}"]
    
    
    ### Allow nn.ModuleList output too??
    return nn.Sequential(layers)

def _stripe_search_indices(n, n_params):
    if length == 1:
        yield (n,)
        return

    for i in range(n + 1):
        for t in get_tuples(n_params - 1, n - i):
            yield (i,) + t
  
def _find_layer_params(in_shape, out_shape, dilation):
    for n in range(8):  # TBD make it per dim, make it accept user defined values
        for padding, kernel, stride in _stripe_search_indices(n, 4):
            if map(lambda x: _size_fcn(x, padding, dilation, kernel + 1, stride + 1), in_shape) == out_shape:
                return padding, kernel + 1, stride + 1
    
    raise RuntimeError("Was not able to find suitable parameters for given inputs!")
    
def _get_conv_layer_type(n_dims):
    return (nn.Conv1d, nn.Conv2d, nn.Conv3d)[n_dims - 1]
    
def _get_pool_layer_type(pool_type, n_dims):
    if pool_type.lower() == "max":
        return (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)[n_dims - 1]
    elif pool_type.lower() == "avg":
        return (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)[n_dims - 1]
    else:
        raise ValueError(f"Got pooling layer type '{pool_type}' not 'max' or 'avg'.")
    
def _get_norm_layer_type(norm_type, n_dims):
    if norm_type == None:
        return None
    elif norm_type.lower() == "batch":
        return (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)[n_dims - 1]
    else:
        raise ValueError(f"Got pooling layer type '{pool_type}' not 'max' or 'avg'.")
    
def _get_dim_sizes(in_size, out_size, n_layers):
    dim_factor = log(out_size / in_size, n_layers)
    dim_sizes = [int(in_size * (channel_factor ** n)) for n in range(1, n_layers+1)]
    return channel_sizes
    
    
def _check_shapes(in_shape, out_shape):
    # must be same dimensionality
    if len(in_shape) != len(out_shape):
        raise ValueError("Input and output shapes must be of the same number of dimensions.")
    # must all be > 0
    if any(map(lambda x : x < 1, in_shape)) or any(map(lambda x : x < 1, out_shape)):
        raise ValueError("Input and output shapes must consist of entries > 0")
    
    return True

def _size_fcn(inp, padding, dilation, kernel, stride):
    numerator = inp + (2*padding) - (dilation*(kernel-1)) - 1
    return int((numerator/stride) + 1)
    
    