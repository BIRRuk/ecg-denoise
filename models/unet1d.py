import torch
# import torchvision.transforms.functional
from torch import nn


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        return x


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


def center_crop1d(x:torch.Tensor, sze:int):
    if x.shape[-1]!=sze:
        start = (x.shape[-1] - sze)//2
        x = x[..., start:start+sze]
    return x


class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        contracting_x = center_crop1d(contracting_x, x.shape[-1])
        x = torch.cat([x, contracting_x], dim=1)
        return x


class UNet(nn.Module):
    # def __init__(self, in_channels: int, out_channels: int, arc = [64, 128, 256, 512, 1024]):
    def __init__(self, in_channels: int, out_channels: int, arc = [32, 64, 128, 256, 512]):
        super().__init__()
        arc_ = list(reversed(arc))
        self.down_conv = nn.ModuleList([
            DoubleConvolution(i, o) for i, o in [
                # (in_channels, arc[0]), (arc[0], arc[1]), (arc[1], arc[2]), (arc[2], arc[3])
                (([in_channels]+arc)[i], ([in_channels]+arc)[i+1]) for i in range(len(arc)-1)
            ]
        ])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.middle_conv = DoubleConvolution(arc[3], arc[4])
        self.up_sample = nn.ModuleList([
            UpSample(i, o) for i, o in [(arc_[i], arc_[i+1]) for i in range(len(arc)-1)]
        ])
        self.up_conv = nn.ModuleList([
            DoubleConvolution(i, o) for i, o in [(arc_[i], arc_[i+1]) for i in range(len(arc)-1)]
        ])
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        self.final_conv = nn.Conv1d(arc[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        pass_through = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)

        x = self.middle_conv(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)

        x = self.final_conv(x)

        return x

if __name__ == '__main__':
    net = UNet(1,1)
    print(net)
    print(net(torch.rand(2,1,1250)))
    