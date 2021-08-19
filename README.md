# pytorch-convo
 A package to automatically set up simple convolutional neural networks in pytorch.
 
## `convo.make_convolutions()`

This is the only function exported by this one-module package, intended to make designing 'quick and dirty'
convolutional sub-networks easy and quick!

Give it the shapes and activations you want, it'll give you a network taking your `in_shape` to your `out_shape`
in `n_layers` steps.

The solver is currently a little rigid, and may fail for larger input sizes and numbers of layers.

### Signature:
```python
convo.make_convolutions(
		   in_shape, out_shape, n_layers, kernel_size=None, stride=None,
                   padding_mode="zeros", dilation=1, bias=True, 
                   activation=nn.ReLU, pool_type="max", norm_type=None, module_list=False
		  ) -> torch.nn.Sequential or torch.nn.ModuleList
```

### Args:

- `in_shape`: A tuple of ints specifying the shape of the input to the convolutional
system. `(C, X, [Y], [Z])`

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
 
- `bias`: Bool or array of Bool. Whether the convolutional layers will use a bias tensor or not.

- `activation`: None or class inheriting torch.nn.Module. One instance of this type will be 
added after each convolutional layer. n.b. this needs to be a class object, NOT an instance
that class (`activation=nn.ReLU` and not `activation=nn.ReLU()`).

- `pool_type`: 'max' or 'avg'. Default 'max'. Indicates whether MaxPool or AvgPool layers will be used.

- `norm_type`: None or 'batch'. Default None. Indicates whether BatchNorm layers will be added after
each pooling layer. In the future other norm types will be implemented.

- `module_list`: Bool. Whether the returned object will be an instance of `torch.nn.Sequential`
or `torch.nn.ModuleList`.

### Returns:

A system of (`n_layers` of) convolutional, activation, pooling and (optionally) norm layers 
taking an input of shape `(batch_size, *in_shape)` and returning a result of shape `(batch_size, *out_shape)`.
These layers are contained in a `torch.nn.Sequential` object or a `torch.nn.ModuleList` if specified by the
`module_list` argument.

## Using convo

Currently, convo is in early testing and has not been released on PyPI. However, feel free to download convo.py
and add it to your project directory to import from locally. You will find bugs so please make PRs or post issues
here so I can get them squashed.

*Also check out my other baby, QutiePy - a python quantum computing library.*

### Example

Below is an example of how to use `make_convolutions` to automate convolution parameter calculations and
keep network definitions tidy. Remember, if you want to check the parameters `make_convolutions` has suggested,
just `print` the resulting `nn.Sequential` or `nn.ModuleList` for a nice representation of the subnetwork in the console.

```python
import torch.nn as nn
from torch import flatten
import convo

class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.conv_subnet = convo.make_convolutions([3, 256, 256], [100, 1, 1], 3)
		## Uncomment the following to inspect the subnet params:
		# print(self.conv_subnet)
		self.flatten = nn.Flatten()
		self.full_conn = nn.Linear(100, 1)
		self.activation = nn.Sigmoid()
	
	def forward(self, batch):
		# batch has shape [BATCH_SIZE, 3, 256, 256]
		flat = self.flattten(self.conv_subnet(batch))
		output = self.activation(self.full_conn(flat))
```

## TODO:

- Make proper docs.
- Test, test, test...
- Automate testing.
- Add more norm layer type.
- Add flexible solver by changing intermediate layer sizes.

