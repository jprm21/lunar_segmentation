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


class UNetMobileNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()

        # ---------- Encoder (MobileNetV2) ----------
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )

        self.encoder = backbone.features

        # Canales reales de MobileNetV2 en puntos clave
        self.enc_channels = [16, 24, 32, 96, 1280]

        # ---------- Decoder ----------
        self.up4 = ConvBlock(self.enc_channels[4] + self.enc_channels[3], 512)
        self.up3 = ConvBlock(512 + self.enc_channels[2], 256)
        self.up2 = ConvBlock(256 + self.enc_channels[1], 128)
        self.up1 = ConvBlock(128 + self.enc_channels[0], 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Guardar tamaño original (H, W)
        input_size = x.shape[-2:]

        # ---------- Encoder ----------
        e1 = self.encoder[0:2](x)     # 1/2
        e2 = self.encoder[2:4](e1)    # 1/4
        e3 = self.encoder[4:7](e2)    # 1/8
        e4 = self.encoder[7:14](e3)   # 1/16
        e5 = self.encoder[14:](e4)    # 1/32

        # ---------- Decoder ----------
        d4 = F.interpolate(e5, scale_factor=2, mode="bilinear", align_corners=False)
        d4 = self.up4(torch.cat([d4, e4], dim=1))

        d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.up3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.up2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([d1, e1], dim=1))

        out = self.classifier(d1)

        
        if out.shape[-2:] != input_size:
            out = F.interpolate(
                out,
                size=input_size,
                mode="bilinear",
                align_corners=False
            )

        return out
