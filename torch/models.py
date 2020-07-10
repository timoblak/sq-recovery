import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class RotationHead(nn.Module):
    def __init__(self, in_features, dense=False, dense_features=64):
        super(RotationHead, self).__init__()

        self.dense = dense
        
        if dense:
            self.out_layer_inter = nn.Linear(in_features, dense_features)
            self.relu = nn.LeakyReLU()
            in_features = dense_features

        self.out_layer = nn.Sequential(
            nn.Linear(in_features, 4)
        )

    def forward(self, x):
        if self.dense:
            x = self.relu(self.out_layer_inter(x))
        x = self.out_layer(x)

        # Normalize quaternion
        x = x / torch.norm(x, 2, -1, keepdim=True)

        return x

class SizeHead(nn.Module):
    def __init__(self, in_features, dense=False):
        super(SizeHead, self).__init__()

        self.dense = dense
        
        if dense:
            self.out_layer_inter = nn.Linear(in_features, dense_features)
            self.relu = nn.LeakyReLU()
            in_features = dense_features

        self.out_layer = nn.Sequential(
            nn.Linear(in_features, 3)
        )

    def forward(self, x):
        if self.dense:
            x = self.relu(self.out_layer_inter(x))
        x = self.out_layer(x)
        x = torch.sigmoid(x)
        return x

class ShapeHead(nn.Module):
    def __init__(self, in_features, dense=False):
        super(ShapeHead, self).__init__()

        self.dense = dense
        
        if dense:
            self.out_layer_inter = nn.Linear(in_features, dense_features)
            self.relu = nn.LeakyReLU()
            in_features = dense_features

        self.out_layer = nn.Sequential(
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        if self.dense:
            x = self.relu(self.out_layer_inter(x))
        x = self.out_layer(x)
        x = torch.sigmoid(x)
        return x

class PositionHead(nn.Module):
    def __init__(self, in_features, dense=False):
        super(PositionHead, self).__init__()

        self.dense = dense
        
        if dense:
            self.out_layer_inter = nn.Linear(in_features, dense_features)
            self.relu = nn.LeakyReLU()
            in_features = dense_features

        self.out_layer = nn.Sequential(
            nn.Linear(in_features, 3)
        )

    def forward(self, x):
        if self.dense:
            x = self.relu(self.out_layer_inter(x))
        x = self.out_layer(x)
        x = torch.sigmoid(x)
        return x

class BlockHead(nn.Module):
    def __init__(self, in_features, dense=False):
        super(BlockHead, self).__init__()

        self.dense = dense
        
        if dense:
            self.out_layer_inter = nn.Linear(in_features, dense_features)
            self.relu = nn.LeakyReLU()
            in_features = dense_features

        self.out_layer = nn.Sequential(
            nn.Linear(in_features, 8)
        )

    def forward(self, x):
        if self.dense:
            x = self.relu(self.out_layer_inter(x))
        x = self.out_layer(x)

        return x

class GenericNetSQ(nn.Module):
    def __init__(self, outputs, fcn=256, dropout=0):
        super(GenericNetSQ, self).__init__()
        # Parameters
        self.outputs = outputs
        self.fcn = fcn
        self.dropout = dropout
        # Convolutions

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=(3, 3)), nn.BatchNorm2d(32), nn.LeakyReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1)), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1)), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=(1, 1)), nn.BatchNorm2d(32), nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(1, 1)), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=(1, 1)), nn.BatchNorm2d(64), nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1, 1)), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1, 1)), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=(1, 1)), nn.BatchNorm2d(128), nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(1, 1)), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1, 1)), nn.BatchNorm2d(256), nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=(1, 1)), nn.BatchNorm2d(256), nn.LeakyReLU()
        )

        self.encoder_fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, self.fcn), nn.LeakyReLU(),
            nn.Linear(self.fcn, self.fcn), nn.LeakyReLU()
        )

        self.output = RotationHead(self.fcn)

    def forward(self, x):
        # Graph

        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        x = self.encoder_fc(x)
        x = self.output.forward(x)

        return x


class ResNetSQ(nn.Module):
    def __init__(self, outputs, fcn=256, dropout=0, pretrained=True):
        super(ResNetSQ, self).__init__()
        # Parameters
        self.outputs = outputs
        self.fcn = fcn
        self.dropout = dropout
        # Convolutions

        self.encoder = models.resnet18(pretrained)

        # Pretrained to grayscale
        self.encoder.conv1.weight = nn.Parameter(torch.sum(self.encoder.conv1.weight, dim=1, keepdim=True))

        self.encoder.fc = nn.Sequential(
            nn.Linear(512, self.fcn), nn.LeakyReLU(),
            nn.Linear(self.fcn, self.fcn), nn.LeakyReLU()
        )
        
        
        self.output_size = SizeHead(self.fcn)
        self.output_shape = ShapeHead(self.fcn)
        self.output_position = PositionHead(self.fcn)
        self.output_rotation = RotationHead(self.fcn)

    def forward(self, x):
        # Graph

        x = self.encoder(x)

        x_size = self.output_size.forward(x)
        x_shape = self.output_shape.forward(x)
        x_position = self.output_position.forward(x)
        x_rotation = self.output_rotation.forward(x)

        return  x_size, x_shape, x_position, x_rotation