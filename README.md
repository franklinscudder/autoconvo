# pytorch-convo
 A package to automatically set up simple convolutional neural networks in pytorch.
 
## ```make_convolutions()```

### Signature:
```python
make_convolutions(in_shape, out_shape, n_layers, kernel_size=None, stride=None,
                        padding_mode="zeros", dilation=1, bias=True, 
                        activation=nn.ReLU, pool_type="max", norm_type=None, module_list=False):
```

### Args:

- `in_shape`: A tuple of ints specifying the shape of the input to the convolutional
system. (C, X, \[Y], \[Z])

- `out_shape`: A tuple with the same shape as in_shape, specifying the desired
output shape. 

- `n_layers`: int, the number of convolutional layers in the system.

- `kernel_size`: None, int, tuple of int (one int per layer) or tuple of tuples of int
 (shape \[n_layers, n_spatial_dims], one int per dimension per layer). The kernel size
 for each convolutional filter. See PyTorch docs for more detail. None means the solver will
 find an appropriate value itself.
 
- `stride`: None, int, tuple of int (one int per layer) or tuple of tuples of int
 (shape \[n_layers, n_spatial_dims], one int per dimension per layer). The stride
 of each convolutional filter. See PyTorch docs for more detail. None means the solver will
 find an appropriate value itself.
 
- `padding_mode`: one of 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'.
	The type of padding used where neccessary. See PyTorch docs for more detail.
	
- `dilation`: None, int, tuple of int (one int per layer) or tuple of tuples of int
 (shape \[n_layers, n_spatial_dims], one int per dimension per layer). The dilation for each 
 convolutional filter. See PyTorch docs for more detail. None means the solver will
 find an appropriate value itself.
 
- `bias`: Bool. Whether the convolutional layers will use a bias tensor or not.

- `activation`: None or class inheriting torch.nn.Module. One instance of this type will be 
added after each convolutional layer. n.b. this needs to be a class object, NOT an instance
that class (`activation=nn.ReLU` and not `activation=nn.ReLU()`).

- `pool_type`: 'max' or 'avg'. Default 'max'. Indicates whether MaxPool or AvgPool layers will be used.

- `norm_type`: None or 'batch'. Default None. Indicates whether BatchNorm layers will be added after
each pooling layer. In the future other norm types will be implemented.

- `module_list`: Bool. Whether the returned object will be an instance of torch.nn.Sequential
or torch.nn.ModuleList.

### Returns

A system of (n_layers \*) convolutional, activation, pooling and (optionally) norm layers 
taking an input of shape (batch_size, \*in_shape) and returning a result of shape (batch_size, \*out_shape).
These layers are contained in a torch.nn.Sequential object or a torch.nn.ModuleList if specified by the
`module_list` argument.



