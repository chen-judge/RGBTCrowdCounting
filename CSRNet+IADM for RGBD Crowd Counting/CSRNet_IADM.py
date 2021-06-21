import torch.nn as nn
import torch
import torch.nn.functional as F


class FusionCSRNet(nn.Module):
    def __init__(self, pretrained=False, ratio=0.7):
        super(FusionCSRNet, self).__init__()
        self.seen = 0

        self.block1 = Block([int(64*ratio), int(64*ratio), 'M'], first_block=True)
        self.block2 = Block([int(128*ratio), int(128*ratio), 'M'], in_channels=int(64*ratio))
        self.block3 = Block([int(256*ratio), int(256*ratio), int(256*ratio), 'M'], in_channels=int(128*ratio))
        self.block4 = Block([int(512*ratio), int(512*ratio), int(512*ratio)], in_channels=int(256*ratio))

        self.backend_feat = [int(512*ratio), int(512*ratio), int(512*ratio), int(256*ratio), int(128*ratio), 64]
        self.backend = make_layers(self.backend_feat, in_channels=int(512*ratio), d_rate=2)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if pretrained:
            self._initialize_weights(mode='normal')
        else:
            self._initialize_weights(mode='kaiming')
                
    def forward(self, RGBT):
        RGB = RGBT[0]
        T = RGBT[1]

        RGB, T, shared = self.block1(RGB, T, None)
        RGB, T, shared = self.block2(RGB, T, shared)
        RGB, T, shared = self.block3(RGB, T, shared)
        _, _, shared = self.block4(RGB, T, shared)

        fusion_feature = shared

        fusion_feature = self.backend(fusion_feature)
        map = self.output_layer(fusion_feature)

        return map

    def _initialize_weights(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if mode == 'normal':
                    nn.init.normal_(m.weight, std=0.01)
                elif mode == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Block(nn.Module):
    def __init__(self, cfg, in_channels=3, first_block=False, dilation_rate=1):
        super(Block, self).__init__()
        self.seen = 0
        self.first_block = first_block
        self.d_rate = dilation_rate

        if first_block is True:
            rgb_in_channels = 3
            t_in_channels = 1
        else:
            rgb_in_channels = in_channels
            t_in_channels = in_channels

        self.rgb_conv = make_layers(cfg, in_channels=rgb_in_channels, d_rate=self.d_rate)
        self.t_conv = make_layers(cfg, in_channels=t_in_channels, d_rate=self.d_rate)
        if first_block is False:
            self.shared_conv = make_layers(cfg, in_channels=in_channels, d_rate=self.d_rate)

        channels = cfg[0]
        self.rgb_msc = MSC(channels)
        self.t_msc = MSC(channels)
        if first_block is False:
            self.shared_fuse_msc = MSC(channels)
        self.shared_distribute_msc = MSC(channels)

        self.rgb_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_fuse_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.rgb_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_distribute_1x1conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, RGB, T, shared):
        RGB = self.rgb_conv(RGB)
        T = self.t_conv(T)
        if self.first_block:
            shared = torch.zeros(RGB.shape).cuda()
        else:
            shared = self.shared_conv(shared)

        new_RGB, new_T, new_shared = self.fuse(RGB, T, shared)
        return new_RGB, new_T, new_shared

    def fuse(self, RGB, T, shared):

        RGB_m = self.rgb_msc(RGB)
        T_m = self.t_msc(T)
        if self.first_block:
            shared_m = shared  # zero
        else:
            shared_m = self.shared_fuse_msc(shared)

        rgb_s = self.rgb_fuse_1x1conv(RGB_m - shared_m)
        rgb_fuse_gate = torch.sigmoid(rgb_s)
        t_s = self.t_fuse_1x1conv(T_m - shared_m)
        t_fuse_gate = torch.sigmoid(t_s)
        new_shared = shared + (RGB_m - shared_m) * rgb_fuse_gate + (T_m - shared_m) * t_fuse_gate

        new_shared_m = self.shared_distribute_msc(new_shared)
        s_rgb = self.rgb_distribute_1x1conv(new_shared_m - RGB_m)
        rgb_distribute_gate = torch.sigmoid(s_rgb)
        s_t = self.t_distribute_1x1conv(new_shared_m - T_m)
        t_distribute_gate = torch.sigmoid(s_t)
        new_RGB = RGB + (new_shared_m - RGB_m) * rgb_distribute_gate
        new_T = T + (new_shared_m - T_m) * t_distribute_gate

        return new_RGB, new_T, new_shared


class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = F.interpolate(self.pool1(x), x.shape[2:])
        x2 = F.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)  # (1, 3C, H, W)
        fusion = self.conv(concat)

        return fusion


def make_layers(cfg, in_channels=3, batch_norm=False, d_rate=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
