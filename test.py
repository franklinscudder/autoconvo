import convo
from torch import ones

in_shape = [3,256,256]
out_shape = [20,1,1]

layers = convo.make_convolutions(in_shape, out_shape, 3, kernel_size=None, 
        stride=None, padding_mode="zeros", dilation=1, bias=True,
        activation=nn.ReLU, pool_type="max", norm_type=None, module_list=False)
print(layers)

print("The following should be the same:")
print(list(layers(ones([1] + in_shape)).shape)[1:])
print(out_shape)
