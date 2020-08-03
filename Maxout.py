import torch.nn as nn

class Maxout(nn.Module):
    def __init__(self, in_features, out_features, num_pieces):
        super(Maxout, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces

        self.lin = nn.Linear(in_features, out_features * num_pieces)

    def forward(self, x):
        y = self.lin(x)
        return y.view(-1, self.out_features).max(dim=1)
