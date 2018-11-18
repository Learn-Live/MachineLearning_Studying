r"""
    learn loss.backward in pytorch
"""
import torch
from torch import nn, optim

torch.manual_seed(1)


def learn_backward():
    input_data = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    lay_1 = nn.Linear(in_features=3, out_features=2)
    lay_2 = nn.Linear(in_features=2, out_features=1)

    def print_layer_parameters(lay):
        for name, params in lay.named_parameters():
            print(f'{name}, {params}')

    # params=torch.Tensor([lay_1.parameters(), lay_2.parameters()])
    params = [{'params': lay_1.parameters()},
              {'params': lay_2.parameters()}]
    optimizer = optim.SGD(params, lr=1e-3)

    for i in range(3):
        optimizer.zero_grad()
        print(f'i={i}')
        z1 = lay_1(input_data)
        print_layer_parameters(lay_1)
        z2 = lay_2(z1)
        print_layer_parameters(lay_2)

        loss = torch.sum(z2)
        print(f'loss={loss}\n')
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    learn_backward()
