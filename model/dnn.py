import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_features, out_features, hid_features, num_layers):
        super(Model, self).__init__()

        self.Linear1 = nn.Linear(in_features, hid_features)
        self.Linearlist = nn.ModuleList([
            nn.Linear(hid_features, hid_features)
        for _ in range(num_layers)
        ])
        self.Linear2 = nn.Linear(hid_features, out_features)

    def forward(self, vec):
        """
            vec (N, C)
        """
        vec = vec.unsqueeze(0)
        out = self.Linear1(vec)
        for net in self.Linearlist:
            out = net(out)
        out = self.Linear2(out)
        out = F.log_softmax(out, dim=1)

        return out

        