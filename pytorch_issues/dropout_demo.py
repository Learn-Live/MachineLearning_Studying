import torch
import torch.nn as nn

m = nn.Dropout()  # default p = 0.5
n = torch.randn(2, 10)

print(m(n))
