import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, dilations=(1, 6, 12, 18)):
        super().__init__()

        branches = []
        for dilation in dilations:
            kernel_size = 1 if dilation == 1 else 3
            padding = 0 if dilation == 1 else dilation
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * len(dilations), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.project(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        psi = self.psi(self.W_g(g) + self.W_x(x))
        return x * psi


class UNetMobileNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        self.encoder = backbone.features

        self.enc_channels = [16, 24, 32, 96, 1280]

        # Bottleneck multiescala ligero
        bottleneck_ch = 256
        self.aspp = ASPP(self.enc_channels[4], out_ch=bottleneck_ch)

        # Attention gates para filtrar skip-connections
        self.att4 = AttentionGate(bottleneck_ch, self.enc_channels[3], 64)
        self.att3 = AttentionGate(512, self.enc_channels[2], 32)
        self.att2 = AttentionGate(256, self.enc_channels[1], 16)
        self.att1 = AttentionGate(128, self.enc_channels[0], 8)

        # Decoder
        self.up4 = ConvBlock(bottleneck_ch + self.enc_channels[3], 512)
        self.up3 = ConvBlock(512 + self.enc_channels[2], 256)
        self.up2 = ConvBlock(256 + self.enc_channels[1], 128)
        self.up1 = ConvBlock(128 + self.enc_channels[0], 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder
        e1 = self.encoder[0:2](x)   # 1/2
        e2 = self.encoder[2:4](e1)  # 1/4
        e3 = self.encoder[4:7](e2)  # 1/8
        e4 = self.encoder[7:14](e3)  # 1/16
        e5 = self.encoder[14:](e4)  # 1/32

        # Bottleneck enriquecido
        e5 = self.aspp(e5)

        # Decoder + attention
        d4 = F.interpolate(e5, scale_factor=2, mode="bilinear", align_corners=False)
        d4 = self.up4(torch.cat([d4, self.att4(d4, e4)], dim=1))

        d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.up3(torch.cat([d3, self.att3(d3, e3)], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.up2(torch.cat([d2, self.att2(d2, e2)], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([d1, self.att1(d1, e1)], dim=1))

        out = self.classifier(d1)

        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        return out
