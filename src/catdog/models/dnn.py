"""Final all-MLP classifier over non-overlapping image patches."""

from torch import Tensor, nn


class MixerBlock(nn.Module):
    def __init__(self, tokens: int, width: int):
        super().__init__()
        self.token_norm = nn.LayerNorm(width)
        #让不同位置的图像块彼此交换信息
        self.token_mlp = nn.Sequential(
            nn.Linear(tokens, tokens), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(tokens, tokens),
        )
        #Channel MLP 在每个图像块内部混合特征
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(width * 2, width), nn.Dropout(0.3),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.token_mlp(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.channel_mlp(x)


class DNNClassifier(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        if image_size % 8:
            raise ValueError("image_size must be divisible by 8")
        self.patch_size = 8 #定义patch大小
        self.grid_size = image_size // 8 #水平和竖直上面可以划分为patch的数量
        tokens = self.grid_size ** 2 #总共patch数量
        self.projection = nn.Linear(3 * 8 * 8, 128)
        self.blocks = nn.Sequential(*(MixerBlock(tokens, 128) for _ in range(4)))
        self.norm = nn.LayerNorm(128)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(128, 2))

    def forward(self, images: Tensor) -> Tensor:
        n = images.shape[0]
        g, p = self.grid_size, self.patch_size
        patches = images.reshape(n, 3, g, p, g, p).permute(0, 2, 4, 1, 3, 5)
        tokens = self.projection(patches.reshape(n, g * g, 3 * p * p))
        return self.classifier(self.norm(self.blocks(tokens)).mean(dim=1))


if __name__ == "__main__":
    model = DNNClassifier(256)
    print(model)