import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, ip=1, hidden=128, op=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(ip, hidden, 4, 2, 1)
        self.conv2 = nn.Conv2d(hidden, hidden * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(hidden * 2, hidden * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(hidden * 4, hidden * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(hidden * 8, op, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.bn2 = nn.BatchNorm2d(hidden * 2)
        self.bn3 = nn.BatchNorm2d(hidden * 4)
        self.bn4 = nn.BatchNorm2d(hidden * 8)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv5(x)
        x = torch.sigmoid(x)
        return x


class Generator(nn.Module):
    def __init__(self, ip=100, hidden=128, op=1):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(ip, hidden * 8, 4, 1, 0)
        self.deconv2 = nn.ConvTranspose2d(hidden * 8, hidden * 4, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(hidden * 2, hidden, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(hidden, op, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(hidden * 8)
        self.bn2 = nn.BatchNorm2d(hidden * 4)
        self.bn3 = nn.BatchNorm2d(hidden * 2)
        self.bn4 = nn.BatchNorm2d(hidden)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.deconv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.deconv5(x)
        x = torch.tanh(x)
        return x
