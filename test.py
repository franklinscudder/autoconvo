from convo import *
from torch import ones

layers = make_convolutions([3,128,128], [20,1,1], 10, kernel_size=None, 
        stride=None, padding_mode="zeros", dilation=1, bias=True,
        activation=nn.ReLU, pool_type="max", norm_type=None, module_list=False)
print(layers)

print(layers(ones(1, 3, 5, 5)).shape)