import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class SPPF(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(SPPF, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        spp_layers = [F.adaptive_max_pool2d(x, output_size=(size, size)) for size in self.pool_sizes]
        spp_layers.append(x)  # Original feature map without pooling
        x = torch.cat(spp_layers, dim=1)  # Concatenate along the channel dimension
        return x


class ConcatBlock(nn.Module):
    def __init__(self, in_channels):
        super(ConcatBlock, self).__init__()
        self.in_channels = in_channels

    def forward(self, x_list):
        x = torch.cat(x_list, dim=1)
        return x


class DetectBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectBlock, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.detect_layers = nn.ModuleList([
            nn.Conv2d(int(in_channels[-1] // 2), int(num_classes), kernel_size=1)
        ])

    def forward(self, x):
        for layer in self.detect_layers:
            x = layer(x)
        return x


class C2fBlock(nn.Module):
    def __init__(self, in_channels, use_batch_norm=True):
        super(C2fBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_batch_norm:
            layers.insert(1, nn.BatchNorm2d(in_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Backbone(nn.Module):
    def __init__(self, config, in_channels):
        super(Backbone, self).__init__()
        self.layers = nn.ModuleList()
        for layer_cfg in config['backbone']:
            self.layers.extend(_parse_layer(layer_cfg))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _parse_layer(layer_cfg):
    if layer_cfg[2] == 'Conv':
        return [nn.Conv2d(layer_cfg[3][1], layer_cfg[3][0], kernel_size=3, stride=layer_cfg[3][2], padding=1)
                for _ in range(layer_cfg[1])]
    elif layer_cfg[2] == 'nn.Upsample':
        return [nn.Upsample(scale_factor=layer_cfg[3][1], mode=layer_cfg[3][2])
                for _ in range(layer_cfg[1])]
    elif layer_cfg[2] == 'C2f':
        return [C2fBlock(layer_cfg[3][0], layer_cfg[3][0])
                for _ in range(layer_cfg[1])]
    elif layer_cfg[2] == 'Concat':
        return [ConcatBlock(layer_cfg[3][0])
                for _ in range(layer_cfg[1])]
    elif layer_cfg[2] == 'Detect':
        return [DetectBlock(layer_cfg[0], layer_cfg[3][0])
                for _ in range(layer_cfg[1])]  # Update to use num_classes from the config
    elif layer_cfg[2] == 'SPPF':
        return [SPPF(layer_cfg[3][1], layer_cfg[3][0]) for _ in range(layer_cfg[1])]


class Head(nn.Module):
    def __init__(self, config):
        super(Head, self).__init__()
        self.layers = nn.ModuleList()
        for layer_cfg in config['head']:
            self.layers.extend(_parse_layer(layer_cfg))

    def forward(self, x_list):
        for layer in self.layers:
            x_list = layer(x_list)
        return x_list


class DModel(nn.Module):
    def __init__(self, config_path):
        super(DModel, self).__init__()
        try:
            with open(config_path) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        except Exception as e:
            raise e

        self.backbone = Backbone(config, in_channels=1)
        self.head = Head(config)

    def forward(self, __x):
        x = self.backbone(__x)
        x = self.head(x)
        return x
