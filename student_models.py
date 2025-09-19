import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_width=128, dropout_rate=0.2):
        super(StudentReconstructiveSubNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.base_width = base_width

        # --- Encoder ---
        self.encoder_conv1 = self._make_conv_block(in_channels, base_width//8, dropout_rate)
        self.encoder_conv2 = self._make_conv_block(base_width//8, base_width//4, dropout_rate)
        self.encoder_conv3 = self._make_conv_block(base_width//4, base_width//2, dropout_rate)
        self.encoder_conv4 = self._make_conv_block(base_width//2, base_width, dropout_rate)

        self.pool = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.bottleneck_conv = self._make_conv_block(base_width, base_width, dropout_rate)

        # --- Decoder ---
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_conv4 = self._make_conv_block(base_width*2, base_width, dropout_rate)
        self.decoder_conv3 = self._make_conv_block(base_width + base_width//2, base_width//2, dropout_rate)
        self.decoder_conv2 = self._make_conv_block(base_width//2 + base_width//4, base_width//4, dropout_rate)
        self.decoder_conv1 = self._make_conv_block(base_width//4 + base_width//8, base_width//8, dropout_rate)

        self.output_conv = nn.Conv2d(base_width//8, out_channels, kernel_size=1)

        # --- 用 1x1 卷積投影瓶頸特徵到教師通道數 ---
        self.bottleneck_proj = nn.Conv2d(base_width, base_width, kernel_size=1)

    def _make_conv_block(self, in_ch, out_ch, dropout_rate):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate>0:
            layers.append(nn.Dropout2d(dropout_rate))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate>0:
            layers.append(nn.Dropout2d(dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        e1 = self.encoder_conv1(x)
        e2 = self.encoder_conv2(self.pool(e1))
        e3 = self.encoder_conv3(self.pool(e2))
        e4 = self.encoder_conv4(self.pool(e3))

        b = self.bottleneck_conv(self.pool(e4))
        b_proj = self.bottleneck_proj(b)  # 投影到教師通道數

        d4 = self.decoder_conv4(torch.cat([self.upsample(b), e4], dim=1))
        d3 = self.decoder_conv3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.decoder_conv2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.decoder_conv1(torch.cat([self.upsample(d2), e1], dim=1))

        out = self.output_conv(d1)

        if return_features:
            return out, b_proj  # 返回重建圖 + 投影後瓶頸特徵
        else:
            return out
