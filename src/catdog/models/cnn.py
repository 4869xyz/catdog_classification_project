"""Final residual CNN classifier."""

from torch import Tensor, nn


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.activation(self.branch(x) + self.skip(x)))


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ResidualConvBlock(3, 32), ResidualConvBlock(32, 64),
            ResidualConvBlock(64, 128), nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 2 * 2, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.35), nn.Linear(128, 2),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.features(images))
    
if __name__ == "__main__":
    model = CNNClassifier()
    print(model)