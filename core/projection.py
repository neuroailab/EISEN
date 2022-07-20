import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', kernel_size=3, stride=1, residual=True, bias=True, downsample=False):
        super(ResidualBlock, self).__init__()

        padding = 1 if kernel_size == 3 else 0
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, bias=bias, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, bias=bias, padding=padding)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1 or downsample is True:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1 or downsample is True:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1 or downsample is True:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1 or downsample is True:
                self.norm3 = nn.Sequential()

        if stride == 1 and not downsample:
            self.shortcut = None

        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=bias), self.norm3)

        self.residual = residual

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.norm2(self.conv2(y))

        if not self.residual:
            return y

        if self.shortcut is not None:
            x = self.shortcut(x)

        return self.relu(x+y)


class KeyQueryProjection(nn.Module):
    def __init__(self, input_dim, kq_dim, latent_dim, kernel_size=3, norm_fn='batch', downsample=True):
        super(KeyQueryProjection, self).__init__()
        self.conv = nn.Conv2d(input_dim, kq_dim, kernel_size=1, bias=True, padding='same')
        self.key = nn.Sequential(
            ResidualBlock(kq_dim, latent_dim, norm_fn, kernel_size=kernel_size, bias=False, stride=1, residual=True, downsample=downsample),
            nn.Conv2d(latent_dim, kq_dim, kernel_size=1, bias=True, padding='same'))
        self.query = nn.Sequential(
            ResidualBlock(kq_dim, latent_dim, norm_fn, kernel_size=kernel_size, bias=False, stride=1, residual=True, downsample=downsample),
            nn.Conv2d(latent_dim, kq_dim, kernel_size=1, bias=True, padding='same'))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # feature projection
        feats = self.conv(x)  # [B, C, H, W]

        # key & query projection
        keys = self.key(feats)  # [B, C, H, W]
        queries = self.query(feats)  # [B, C, H, W]

        return keys, queries