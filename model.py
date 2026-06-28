import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class SARToOpticalGenerator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = ConvBlock(features, features * 2, down=True, act="leaky")
        self.down2 = ConvBlock(features * 2, features * 4, down=True, act="leaky")
        self.down3 = ConvBlock(features * 4, features * 8, down=True, act="leaky")
        self.down4 = ConvBlock(features * 8, features * 8, down=True, act="leaky")
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = ConvBlock(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = ConvBlock(features * 16, features * 4, down=False, act="relu")
        self.up3 = ConvBlock(features * 8, features * 2, down=False, act="relu")
        self.up4 = ConvBlock(features * 4, features, down=False, act="relu")
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        bn = self.bottleneck(d5)
        
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        return self.final_up(torch.cat([u4, d2], dim=1))

if __name__ == "__main__":
    x = torch.randn((1, 1, 256, 256))
    model = SARToOpticalGenerator()
    print("Output tensor matrix validation shape:", model(x).shape)
