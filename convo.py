"""
convo.py

A package to automatically set up simple convolutional neural networks in pytorch.

Thomas Findlay, 08/2021

findlaytel@gmail.com
"""

import torch.nn as nn
from torch import tensor, zeros
from collections import OrderedDict
from math import log, exp, ceil, floor
import numpy as np

class FailedToSolve(Exception):
    pass

class _LastUpdatedOrderedDict(OrderedDict):
    """Store items in the order the keys were last added"""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        

def make_convolutions(in_shape, out_shape, n_layers, kernel_size=None, stride=None,
                        padding_mode="zeros", dilation=1, bias=True, 
                        activation=nn.ReLU, pool_type="max", norm_type=None, module_list=False):
    """Return a convolutional subnetwork configured according to the given args.

    Parameters
    ----------
    
    in_shape : tuple of ints
        The shape of the input to the convolutional system. (C, X, [Y], [Z]).

    out_shape : tuple of ints 
        With the same shape as in_shape, specifying the desired output shape. 

    n_layers : int
        The number of convolutional layers in the system.

    kernel_size : None, int, tuple of int (one int per layer) or tuple of tuples of int (shape [n_layers, n_spatial_dims], one int per dimension per layer), Default None
        The kernel size for each convolutional filter. See PyTorch docs for more detail. None means the solver will find an appropriate value itself.
 
    stride : None, int, tuple of int (one int per layer) or tuple of tuples of int (shape [n_layers, n_spatial_dims], one int per dimension per layer), Default None
        The stride of each convolutional filter. See PyTorch docs for more detail. None means the solver will find an appropriate value itself.
 
    padding_mode: one of 'zeros', 'reflect', 'replicate' or 'circular', Default 'zeros'.
        The type of padding used where neccessary. See PyTorch docs for more detail.
	
    dilation: int, tuple of int (one int per layer) or tuple of tuples of int (shape [n_layers, n_spatial_dims], one int per dimension per layer), Default 1
        The dilation for each convolutional filter. See PyTorch docs for more detail. None means the solver will find an appropriate value itself.
 
    bias: Bool or array of Bool length n_layers, Default: True
        Whether the convolutional layers will use a bias tensor or not.

    activation: None or class inheriting torch.nn.Module, Default None
        One instance of this type will be added after each convolutional layer. n.b. this needs to be a class object, NOT an instance that class.

    pool_type: 'max', 'avg' or array of both length n_layers, Default 'max'
        Indicates whether MaxPool or AvgPool layers will be used.

    norm_type: None or 'batch' or array of both, Default None
        Indicates whether BatchNorm layers will be added after  each pooling layer. In the future other norm types will be implemented.

    module_list: Bool, Default False
        Whether the returned object will be an instance of torch.nn.Sequential or torch.nn.ModuleList.

    Returns
    -------
    torch.nn.Sequential or torch.nn.ModuleList
        The resulting convolutional subnetwork in the specified container.
        

    Raises
    ------
    RuntimeError
        If the parameter solver fails, try different constraints.
        
    TypeError
        If certain arguments are given as incompatible types.
        
    ValueError
        If the value of certain arguments are not in the correct range or shape.

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
    solved = False
    # modifier for intermediate layer shapes in case solve fails
    for modifier in _get_modifier(n_layers, n_dims):
        #print(modifier)
        try:
            for dim in range(n_dims + 1):
                overall_dim_shapes[:, dim] = tensor(_get_dim_sizes(in_shape[dim], out_shape[dim], n_layers))
    
            ks = kernel_size
            st = stride
            
            #print("before", overall_dim_shapes)
            overall_dim_shapes[1:-1,1:] += modifier
            #print("after", overall_dim_shapes)
            dim_shapes = np.repeat(overall_dim_shapes, 2, axis=0)

            for odds in range(1, 2*n_layers + 1, 2):
                dim_shapes[odds, 0] = dim_shapes[odds, 0]
        
            conv_padding = zeros((n_layers, n_dims))
            conv_kernel_size = zeros((n_layers, n_dims))
            conv_stride = zeros((n_layers, n_dims))
    
            pool_padding = zeros((n_layers, n_dims))
            pool_kernel_size = zeros((n_layers, n_dims))
            pool_stride = zeros((n_layers, n_dims))

            for n in range(n_layers):
                for dim in range(n_dims):
                    conv_padding[n, dim], conv_kernel_size[n, dim], conv_stride[n, dim], \
                    pool_padding[n, dim], pool_kernel_size[n, dim], pool_stride[n, dim] = _get_layer_params(overall_dim_shapes[n, dim+1], overall_dim_shapes[n+1, dim+1], 0, ks[n, dim], st[n,dim])
            
            solved = True
            #print(overall_dim_shapes)
            break
            
        except FailedToSolve:
            pass
     
    if not solved:
        raise RuntimeError("Was not able to find suitable parameters for given inputs!")
    
    conv_layer_type = _get_conv_layer_type(n_dims)
    
    for n in range(n_layers):
        conv_args, conv_kwargs = _get_conv_args(overall_dim_shapes[n, 0], overall_dim_shapes[n+1, 0], conv_kernel_size[n],
                                  conv_stride[n], conv_padding[n,:], dilation[n], bias[n], padding_mode)
        pool_args, pool_kwargs = _get_pool_args(pool_kernel_size[n, :], pool_stride[n, :], pool_padding[n, :])
        norm_args, norm_kwargs = _get_norm_args(overall_dim_shapes[n+1, 0])
        
        pool_layer_type = _get_pool_layer_type(pool_type[n], n_dims)
        norm_layer_type = _get_norm_layer_type(norm_type[n], n_dims)
        activation_layer_type = activation[n]
        
        layers[f"conv_{n + 1}"] = conv_layer_type(*conv_args, **conv_kwargs)
        
        if activation is not None:
            layers[f"activation_{n + 1}"] = activation_layer_type()
        
        layers[f"pool_{n + 1}"] = pool_layer_type(*pool_args, **pool_kwargs)
        
        if norm_type[n] is not None:
            layers[f"{str(norm_layer_type).split('.')[1]}_norm_{n + 1}"] = norm_layer_type(*norm_args, **norm_kwargs)
    
    if module_list:
        return nn.ModuleList(layers.values())
    
    return nn.Sequential(layers)

def _get_modifier(n_layers, n_dims):
    
    for depth in range(n_layers * n_dims):
        g = _stripe_search_indices((n_layers-1)*n_dims, depth)
        for m in g:
            yield tensor(m).view(n_layers-1, n_dims)
 
def _get_norm_args(in_channels):
    return (int(in_channels),), {}

def _get_pool_args(kernel, stride, padding):
    return (([int(k) for k in kernel],) ,
            {
            "stride" : [int(s) for s in stride],
            "padding" : [int(p) for p in padding]
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
    """ Swaps None to -1 for torch tensor compatibility. """
    if x is None:
        return -1
    else:
        return x
         
def _parse_dim_arg(arg, arg_name, n_dims):
    """ Parses args that can be per-dimension. """
    try:
        [iter(x) for x in arg]
    except TypeError:
        return tensor([[_none_to_neg_1(x)] * n_dims for x in arg])  # edge case: SOME dims are iters
    
    if len(arg[0]) == n_dims:
        return tensor(arg)
    else:
        raise TypeError(f"{arg_name} must be single value or iterable of length n_layers or array of shape (n_layers, n_dims), got {type(arg)}.") 
        
def _parse_seq_arg(arg, arg_name, n_layers):
    """ Parses args that can be per-layer. """
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
    """ Yields tuples of n_params ints which sum to n_stripe. """
    if n_params == 1:
        yield (n_stripe,)
        return

    for i in range(n_stripe + 1):
        for t in _stripe_search_indices(n_params - 1, n_stripe - i):
            yield (i,) + t

def _double_stripe_indices(n_params, n_stripe):
    """ Encapsulates two _stripe_search_indices generators in parallel, calling next on one then the other."""
    g1 = _stripe_search_indices(ceil(n_params//2), n_stripe)
    g2 = _stripe_search_indices(floor(n_params//2), n_stripe)
    
    g2_last = next(g2)
    
    while 1:
        try:
            g1_last = next(g1)
            #print(n_stripe, g1_last + g2_last)
            yield g1_last + g2_last
            g2_last = next(g2)
            #print(n_stripe, g1_last + g2_last)
            yield g1_last + g2_last
            
        except StopIteration:
            return
  
def _get_layer_params(in_size, out_size, dilation, ks, st):
    """ 
    Solves for the params of a conv-pool layer pair given the in and out sizes 
    of a given dim and use specified kernel and stride values. 
    """
    stride_low_bound = 1
    
    if in_size > 128:
        kernel_low_bound = 3
    elif in_size > 512:
        kernel_low_bound = 5
    else:
        kernel_low_bound = 1
    
    for n in range(13):
        #print("solving")
        for conv_padding, conv_kernel, conv_stride, pool_padding, pool_kernel, pool_stride  in _double_stripe_indices(6, n):
            if ks != -1:
                #print(1)
                conv_kernel, pool_kernel = ks - kernel_low_bound, ks - kernel_low_bound
            
            if st != -1:
                #print(2)
                conv_stride, pool_stride = st - stride_low_bound, st - stride_low_bound
            
            if conv_stride > conv_kernel + kernel_low_bound:
                #print(3)
                continue
            
            if pool_stride > pool_kernel + kernel_low_bound:
                #print(4)
                continue
                
            if 2 * pool_padding > pool_kernel + kernel_low_bound:
                #print(5)
                continue
                
            if in_size + (conv_padding * 2) < conv_kernel + kernel_low_bound:
                #print(6)
                continue
            
            inter_size = _conv_size_fcn(in_size, conv_padding, dilation, conv_kernel + kernel_low_bound, conv_stride + stride_low_bound)
            
            if inter_size + (pool_padding * 2) < pool_kernel:
                #print(7)
                continue
            
            candidate_out_size = _pool_size_fcn(inter_size, pool_padding, pool_kernel + kernel_low_bound, pool_stride + stride_low_bound)
            
            #print(candidate_out_size, out_size)
            #print(conv_padding, conv_kernel + kernel_low_bound, conv_stride + stride_low_bound, pool_padding, pool_kernel + kernel_low_bound, pool_stride + stride_low_bound, "\n")
            
            if candidate_out_size == out_size:
                return conv_padding, conv_kernel + kernel_low_bound, conv_stride + stride_low_bound, pool_padding, pool_kernel + kernel_low_bound, pool_stride + stride_low_bound
            
            #print(8)
            
    raise FailedToSolve("Was not able to find suitable parameters for given inputs!")
    
def _get_conv_layer_type(n_dims):
    """ Returns the type of an n_dims-dimensioinal convolution layer. """
    return (nn.Conv1d, nn.Conv2d, nn.Conv3d)[n_dims - 1]
    
def _get_pool_layer_type(pool_type, n_dims):
    """ Returns the type of an n_dims-dimensioinal pooling layer of the specified type. """
    if pool_type is None:
        return None
    elif pool_type.lower() == "max":
        return (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)[n_dims - 1]
    elif pool_type.lower() == "avg":
        return (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)[n_dims - 1]
    else:
        raise ValueError(f"Got pooling layer type '{pool_type}' not 'max' or 'avg'.")
    
def _get_norm_layer_type(norm_type, n_dims):
    """ Returns the type of an n_dims-dimensioinal norm layer of the specified type. """
    if norm_type is None:
        return None
    elif norm_type.lower() == "batch":
        return (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)[n_dims - 1]
    else:
        raise ValueError(f"Got pooling layer type '{pool_type}' not 'max' 'avg' or None.")
    
def _get_dim_sizes(in_size, out_size, n_layers):
    """ 
    Return an approximately geometric series of n_layers+1 ints,
    starting with in_size and ending with out_size.
    """
    # in * fac ** n = out,   log out/in = n log fac,   fac = (out/in) ^ 1/n

    dim_factor = (out_size / in_size) ** (1 / n_layers)
    dim_sizes = [int(in_size * (dim_factor ** n)) for n in range(0, n_layers+1)]
    dim_sizes = [s+(int(s<1)*(1-s)) for s in dim_sizes] # makes all sizes >= 1
    dim_sizes[-1] = out_size
    
    return dim_sizes
    
    
def _check_shapes(in_shape, out_shape):
    """ Check that valid shapes have been supplied by the caller. """
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
    """ Return the output size or a conv layer with the given params. """
    numerator = inp + (2*padding) - (dilation*(kernel-1)) - 1
    return int((numerator/stride) + 1)
    
def _pool_size_fcn(inp, padding, kernel, stride):
    """ Return the output size or a pool layer with the given params. """
    numerator = inp + (2*padding) - kernel
    return int((numerator/stride) + 1)
    
  