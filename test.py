import convo
from torch import ones
import torch.nn as nn

in_shape = [3,312,256]
out_shape = [2000,1,1]

layers = convo.make_convolutions(in_shape, out_shape, 3, kernel_size=[9, None, None], 
        stride=None, padding_mode="zeros", dilation=1, bias=True,
        activation=nn.Tanh, pool_type="max", norm_type="batch", module_list=False)
print(layers)

print("The following should be the same:")
print(list(layers(ones([1] + in_shape)).shape)[1:])
print(out_shape)
