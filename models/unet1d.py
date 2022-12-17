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
        print(x.shape)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.middle_conv = DoubleConvolution(512, 1024)
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        pass_through = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            print(x.shape)
            pass_through.append(x)
            x = self.down_sample[i](x)
            print(x.shape)

        x = self.middle_conv(x)
        print(x.shape)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)
            print(x.shape)

        x = self.final_conv(x)

        return x

if __name__ == '__main__':
    net = UNet(1,1)
    print(net)
    print(net(torch.rand(2,1,1250)))
    