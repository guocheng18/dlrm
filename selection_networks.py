"""
Defined quantization bit selection network and dimension selection network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EdimQbitSelectionNetwork(nn.Module):
    '''
    The network that selects the number of embedding dimensions and quantization bits.
    Input:
        x: the feature vector of the input.
        freq: the frequency of the input.
    '''
    def __init__(self, input_dim: int, hidden_dim: int, num_edims_selections: int, num_qbits_selections: int):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim+1, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim, bias=True),
            nn.ReLU(),
        )
        self.edims_selector = nn.Linear(2*hidden_dim, num_edims_selections, bias=True)
        self.qbits_selector = nn.Linear(2*hidden_dim, num_qbits_selections, bias=True)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias)

    def forward(self, x: torch.Tensor, freq: torch.Tensor):
        x = torch.cat((x, freq.unsqueeze(1)), dim=1)
        x = self.feat(x)
        y1 = F.gumbel_softmax(self.edims_selector(x), tau=1, hard=True)
        y2 = F.gumbel_softmax(self.qbits_selector(x), tau=1, hard=True)
        return y1, y2


class QuantBitSelectionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_selections: int) -> None:
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_selections, bias=True)
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.selector(x)
        y = F.gumbel_softmax(x, tau=1, hard=True)
        return y

class EmbeddingDimensionSelectionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_selections: int) -> None:
        super().__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_selections, bias=True)
        )
        self.init_parameters()

    def init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.selector(x)
        y = F.gumbel_softmax(x, tau=1, hard=True)
        return y

if __name__ == '__main__':
   selector = EdimQbitSelectionNetwork(10, 10, 3, 3) 
   input = torch.randn(1, 10)
   print(selector(input))