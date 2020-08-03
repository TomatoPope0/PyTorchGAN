import torch.nn as nn

class Maxout(nn.Module):
    def __init__(self, in_features, out_features, num_pieces):
        super(Maxout, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces

        self.lin = nn.Linear(in_features, out_features * num_pieces)

    def forward(self, x):
        lin = self.lin(x)
        y = lin.view(-1, self.num_pieces, self.out_features).max(dim=1)[0]
        return y
