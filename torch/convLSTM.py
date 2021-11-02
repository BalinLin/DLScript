import torch
from pytorch_convolutional_rnn import convolutional_rnn
from torch.nn.utils.rnn import pack_padded_sequence

# https://iter01.com/547792.html
# https://github.com/kamo-naoyuki/pytorch_convolutional_rnn
in_channels = 512
net = convolutional_rnn.Conv2dLSTM(in_channels=in_channels,  # Corresponds to input size
                                   out_channels=6,  # Corresponds to hidden size
                                   kernel_size=3,  # Int or List[int]
                                   num_layers=2,
                                   bidirectional=True,
                                   dilation=2, stride=2, dropout=0.5,
                                   batch_first=True)
length = 3
batchsize = 2
shape = (10, 14)
x = torch.randn(batchsize, length, in_channels, *shape)
print("x: ", x.shape)
h = None
y, h = net(x, h)
print("y: ", y.shape) # (batchsize, length, out_channels * 2, *shape)
print("h: ", len(h))  # (2, num_layers * 2, batchsize, out_channels, *shape)
print("h: ", len(h[0]))
print("h: ", len(h[0][0]))
print("h: ", len(h[0][0][0]))
print("h: ", len(h[0][0][0][0]))
print("h: ", len(h[0][0][0][0][0]))
