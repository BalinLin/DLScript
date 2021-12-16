import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

if __name__ == "__main__":
    embedding = nn.Embedding(10, 3)
    # a batch of 2 samples of 4 indices each
    input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    output = embedding(input)
    print(output.shape)

    input = torch.LongTensor([1,2,4,5])
    output = embedding(input)
    print(output.shape)

    embedding = nn.Embedding(51, 32)
    output = embedding(torch.from_numpy(np.arange(0, 51)))
    print(output.shape)
