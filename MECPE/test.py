import random
import torch
from torch import nn
import numpy as np

m = nn.LogSoftmax(dim=-1)
input = torch.randn(2, 3)
output = m(input)
print(output)