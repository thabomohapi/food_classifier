import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

class EfficientMBConv(nn.Module):
    """Memory-efficient MBConv block with squeeze-and-excitation"""
    def __init__(self, in_channels, out_channels, stride=1, expansion_factor=4):
        super().__init__()
        self.residual = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expansion_factor

        # Point-wise expansion
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ) if expansion_factor != 1 else nn.Identity()

        # Depth-wise convolution
        self.dwconv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, 
                     groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )

        # Squeeze-and-excitation
        squeeze_channels = max(1, in_channels // 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, squeeze_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_channels, expanded_channels, 1),
            nn.Sigmoid()
        )

        # Point-wise projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Stochastic depth
        self.drop_path = DropPath(0.2) if self.residual else nn.Identity()

    def forward(self, x):
        identity = x

        # MBConv operations
        out = self.expand_conv(x)
        out = self.dwconv(out)
        
        # Squeeze-and-excitation
        out = out * self.se(out)
        
        # Projection
        out = self.project_conv(out)

        # Residual connection with stochastic depth
        if self.residual:
            out = identity + self.drop_path(out)
        
        return out

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class PyramidFeatureBlock(nn.Module):
    """Feature pyramid block for multi-scale feature extraction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, 
                                  groups=out_channels, bias=False)
        self.bn3x3_1 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, 3, 1, 2, 
                                  dilation=2, groups=out_channels, bias=False)
        self.bn3x3_2 = nn.BatchNorm2d(out_channels)
        
        self.channel_shuffle = ChannelShuffle(groups=4)
        
    def forward(self, x):
        x = F.silu(self.bn1x1(self.conv1x1(x)))
        
        path1 = F.silu(self.bn3x3_1(self.conv3x3_1(x)))
        path2 = F.silu(self.bn3x3_2(self.conv3x3_2(x)))
        
        out = path1 + path2
        out = self.channel_shuffle(out)
        return out

class EnhancedResNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Efficient stem with early feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        # Main network stages with increasing complexity
        self.stage1 = self._make_stage(64, 128, stride=2)
        self.pyramid1 = PyramidFeatureBlock(128, 128)
        
        self.stage2 = self._make_stage(128, 256, stride=2)
        self.pyramid2 = PyramidFeatureBlock(256, 256)
        
        self.stage3 = self._make_stage(256, 512, stride=2)
        self.pyramid3 = PyramidFeatureBlock(512, 512)

        # Global pooling with both max and average features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Advanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, stride):
        return nn.Sequential(
            EfficientMBConv(in_channels, out_channels, stride=stride, expansion_factor=4),
            EfficientMBConv(out_channels, out_channels, stride=1, expansion_factor=4)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 1
        x = self.stage1(x)
        x = self.pyramid1(x)

        # Stage 2
        x = self.stage2(x)
        x = self.pyramid2(x)

        # Stage 3
        x = self.stage3(x)
        x = self.pyramid3(x)

        # Global pooling
        avg_features = self.avg_pool(x)
        max_features = self.max_pool(x)
        x = torch.cat([avg_features, max_features], dim=1)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)
        return x

    def get_embedding(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pyramid1(x)
        x = self.stage2(x)
        x = self.pyramid2(x)
        x = self.stage3(x)
        x = self.pyramid3(x)
        
        avg_features = self.avg_pool(x)
        max_features = self.max_pool(x)
        x = torch.cat([avg_features, max_features], dim=1)
        return x.view(x.size(0), -1)