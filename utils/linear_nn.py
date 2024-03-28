import torch
from torch.autograd import Variable
import numpy as np


class linearNet(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearNet, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

# Main function
if __name__ == '__main__':
    num_actions = 2
    num_features = 55

    net = linearNet(inputSize=num_features, outputSize=num_actions)
    net = net.float()
    print(f"net: {net.linear.weight}")

    state = np.zeros((num_features))
    state[0] = 1

    state = torch.from_numpy(state).float()
    
    print(state)

    q_val = net.forward(state)
    print(q_val)
