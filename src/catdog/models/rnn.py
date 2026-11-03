"""Final row-and-column bidirectional GRU classifier."""

import torch
from torch import Tensor, nn


class RNNClassifier(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        self.row_projection = nn.Sequential(nn.Linear(3 * image_size, 64), nn.LayerNorm(64), nn.GELU())
        self.column_projection = nn.Sequential(nn.Linear(3 * image_size, 64), nn.LayerNorm(64), nn.GELU())
        self.row_rnn = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.column_rnn = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(256), nn.Dropout(0.4), nn.Linear(256, 2),
        )

    def forward(self, images: Tensor) -> Tensor:
        rows = images.permute(0, 2, 1, 3).flatten(start_dim=2)
        columns = images.permute(0, 3, 1, 2).flatten(start_dim=2)
        row_states, _ = self.row_rnn(self.row_projection(rows))
        column_states, _ = self.column_rnn(self.column_projection(columns))
        features = torch.cat((row_states.mean(dim=1), column_states.mean(dim=1)), dim=1)
        return self.classifier(features)

if __name__ =="__main__":
    model = RNNClassifier(256)
    print(model)