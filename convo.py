"""
convo.py

A package to automatically set up simple convolutional neural networks in pytorch.

Thomas Findlay, 08/2021

findlaytel@gmail.com
"""

import torch.nn as nn
from torch import tensor, zeros
from collections import OrderedDict
from math import log, exp
import numpy as np

class _LastUpdatedOrderedDict(OrderedDict):
    """Store items in the order the keys were last added"""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        

def make_convolutions(in_shape, out_shape, n_layers, kernel_size=None, stride=None,
                        padding_mode="zeros", dilation=0, bias=True, 
                        activation=nn.ReLU, pool_type="max", norm_type=None, module_list=False):
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
    activation: nn.Module or None
    """

    _check_shapes(in_shape, out_shape)
    
    n_dims = len(in_shape) - 1 # SPATIAL dims
    
    #can_be_seq = ["kernel_size", "stride", "dilation", "bias", "activation", "norm_type", "pool_type"]
    
    kernel_size = _parse_seq_arg(kernel_size, "kernel_size", n_layers)
    stride = _parse_seq_arg(stride, "stride", n_layers)
    dilation = _parse_seq_arg(dilation, "dilation", n_layers)
    bias = _parse_seq_arg(bias, "bias", n_layers)
    activation = _parse_seq_arg(activation, "activation", n_layers)
    norm_type = _parse_seq_arg(norm_type, "norm_type", n_layers)
    pool_type = _parse_seq_arg(pool_type, "pool_type", n_layers)
    
    #can_be_dim = ["kernel_size", "stride", "dilation"]
    kernel_size = _parse_dim_arg(kernel_size, "kernel_size", n_dims)
    stride = _parse_dim_arg(stride, "stride", n_dims)
    dilation = _parse_dim_arg(dilation, "dilation", n_dims)
    
    layers = _LastUpdatedOrderedDict()
    
    overall_dim_shapes = zeros((n_layers+1, n_dims+1))
    for dim in range(n_dims + 1):
        overall_dim_shapes[:, dim] = tensor(_get_dim_sizes(in_shape[dim], out_shape[dim], n_layers))
    
    ks = kernel_size
    st = stride
    
    dim_shapes = np.repeat(overall_dim_shapes, 2, axis=0)
    print("overall dim shapes:\n", overall_dim_shapes)
    input()
    for odds in range(1, 2*n_layers + 1, 2):
        dim_shapes[odds, 0] = dim_shapes[odds, 0]
        
    conv_padding = zeros((n_layers, n_dims))
    conv_kernel_size = zeros((n_layers, n_dims))
    conv_stride = zeros((n_layers, n_dims))
    
    pool_padding = zeros((n_layers, n_dims))
    pool_kernel_size = zeros((n_layers, n_dims))
    pool_stride = zeros((n_layers, n_dims))
    
    ### TBD change np arrays to Tensors

    for n in range(n_layers):
        for dim in range(n_dims):
            conv_padding[n, dim], conv_kernel_size[n, dim], conv_stride[n, dim], \
            pool_padding[n, dim], pool_kernel_size[n, dim], pool_stride[n, dim] = _get_layer_params(overall_dim_shapes[n, dim], overall_dim_shapes[n+1, dim], 0, ks[n, dim], st[n,dim])
      
    conv_layer_type = _get_conv_layer_type(n_dims)
    
    for n in range(n_layers):
        conv_args, conv_kwargs = _get_conv_args(dim_shapes[n, 0], dim_shapes[n+1, 0], conv_kernel_size[n],
                                        conv_stride[n], conv_padding[n,:], dilation[n], bias[n], padding_mode)
        pool_args, pool_kwargs = _get_pool_args(pool_kernel_size[n, :], pool_stride[n, :], pool_padding[n, :])
        norm_args, norm_kwargs = _get_norm_args(dim_shapes[n+2, 0])
        
        print(conv_args, conv_kwargs)
        
        pool_layer_type = _get_pool_layer_type(pool_type[n], n_dims)
        norm_layer_type = _get_norm_layer_type(norm_type[n], n_dims)
        activation_layer_type = activation[n]
        
        layers[f"conv_{n + 1}"] = conv_layer_type(*conv_args, **conv_kwargs)
        
        if activation is not None:
            layers[f"activation_{n + 1}"] = activation_layer_type()
        
        layers[f"pool_{n + 1}"] = pool_layer_type(*pool_args, **pool_kwargs)
        
        if norm_type[n] is not None:
            print(norm_type, norm_layer_type)
            layers[f"{norm_layer_type}_norm_{n + 1}"] = norm_layer_type(*norm_args, **norm_kwargs)
    
    if module_list:
        return nn.ModuleList(layers.values())
    
    return nn.Sequential(layers)
    
def _get_norm_args(in_channels):
    return (in_channels,), {}

def _get_pool_args(kernel, stride, padding):
    return ((kernel,) ,
            {
            "stride" : stride,
            "padding" : padding
            })
    
def _get_conv_args(in_channels, out_channels, kernel, stride, padding, dilation, bias, padding_mode):
    return ((int(in_channels), int(out_channels), [int(k) for k in kernel]) ,
            {
            "stride" : [int(s) for s in stride],
            "padding" : [int(p) for p in padding],
            "dilation" : [int(d) for d in dilation],
            "bias" : bias,
            "padding_mode" : padding_mode,
            })

def _none_to_neg_1(x):
    if x is None:
        return -1
    else:
        return x
         
def _parse_dim_arg(arg, arg_name, n_dims):
    try:
        [iter(x) for x in arg]
    except TypeError:
        return tensor([[_none_to_neg_1(x)] * n_dims for x in arg])  # edge case: SOME dims are iters
    
    if len(arg[0]) == n_dims:
        return tensor(arg)
    else:
        raise TypeError(f"{arg_name} must be single value or iterable of length n_layers or array of shape (n_layers, n_dims), got {type(arg)}.") 
        
def _parse_seq_arg(arg, arg_name, n_layers):
    if type(arg) == str:
        return [arg] * n_layers
        
    try:
        iter(arg)
    except TypeError:
        return [arg] * n_layers
        
    if len(arg) == n_layers:
        return arg
    else:
        raise TypeError(f"{arg_name} must be single value or iterable of length n_layers, got: {type(arg)}.") 

def _stripe_search_indices(n_params, n_stripe):
    if n_params == 1:
        yield (n_stripe,)
        return

    for i in range(n_stripe + 1):
        for t in _stripe_search_indices(n_params - 1, n_stripe - i):
            yield (i,) + t
  
def _get_layer_params(in_size, out_size, dilation, ks, st):
    stride_low_bound = 1
    
    if in_size > 128:
        kernel_low_bound = 3
    elif in_size > 512:
        kernel_low_bound = 5
    else:
        kernel_low_bound = 1
    
    for n in range(9):
        for conv_padding, conv_kernel, conv_stride, pool_padding, pool_kernel, pool_stride  in _stripe_search_indices(6, n):
            if ks != -1:
                conv_kernel, pool_kernel = ks - kernel_low_bound, ks - kernel_low_bound
            
            if st != -1:
                conv_stride, pool_stride = st - stride_low_bound, st - stride_low_bound
            
            if conv_stride > conv_kernel + kernel_low_bound:
                continue
            
            if pool_stride > pool_kernel + kernel_low_bound:
                continue
            
            inter_size = _conv_size_fcn(in_size, conv_padding, dilation, conv_kernel + kernel_low_bound, conv_stride + stride_low_bound)
            candidate_out_size = _pool_size_fcn(inter_size, pool_padding, pool_kernel + kernel_low_bound, pool_stride + stride_low_bound)
            
            print(candidate_out_size, out_size)
            if candidate_out_size == out_size:
                print("Solution:", conv_padding, conv_kernel + kernel_low_bound, conv_stride + stride_low_bound, pool_padding, pool_kernel + kernel_low_bound, pool_stride + stride_low_bound)
                return conv_padding, conv_kernel + kernel_low_bound, conv_stride + stride_low_bound, pool_padding, pool_kernel + kernel_low_bound, pool_stride + stride_low_bound
    
    raise RuntimeError("Was not able to find suitable parameters for given inputs!")
    
def _get_conv_layer_type(n_dims):
    return (nn.Conv1d, nn.Conv2d, nn.Conv3d)[n_dims - 1]
    
def _get_pool_layer_type(pool_type, n_dims):
    if pool_type is None:
        return None
    elif pool_type.lower() == "max":
        return (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)[n_dims - 1]
    elif pool_type.lower() == "avg":
        return (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)[n_dims - 1]
    else:
        raise ValueError(f"Got pooling layer type '{pool_type}' not 'max' or 'avg'.")
    
def _get_norm_layer_type(norm_type, n_dims):
    if norm_type is None:
        return None
    elif norm_type.lower() == "batch":
        return (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)[n_dims - 1]
    else:
        raise ValueError(f"Got pooling layer type '{pool_type}' not 'max' 'avg' or None.")
    
def _get_dim_sizes(in_size, out_size, n_layers):
    # in * fac ** n = out,   log out/in = n log fac
    
    # OPTIMIZE ME
    dim_factor = exp(log(out_size / in_size) / n_layers)
    dim_sizes = [int(in_size * (dim_factor ** n)) for n in range(0, n_layers+1)]
    return dim_sizes
    
    
def _check_shapes(in_shape, out_shape):
    try:
        iter(in_shape)
        iter(out_shape)
    except:
        raise TypeError("Shape values must be iterables.")
    if len(in_shape) not in (2, 3, 4):
        raise ValueError("Shapes must be of length 2, 3 or 4 (c, x, [y], [z]).")
    # must be same dimensionality
    if len(in_shape) != len(out_shape):
        raise ValueError("Input and output shapes must be of the same number of dimensions.")
    # must all be > 0
    if any(map(lambda x : x < 1, in_shape)) or any(map(lambda x : x < 1, out_shape)):
        raise ValueError("Input and output shapes must consist of entries > 0")
    
    return True

def _conv_size_fcn(inp, padding, dilation, kernel, stride):
    numerator = inp + (2*padding) - (dilation*(kernel-1)) - 1
    return int((numerator/stride) + 1)
    
def _pool_size_fcn(inp, padding, kernel, stride):
    numerator = inp + (2*padding) - kernel
    return int((numerator/stride) + 1)
    
    