import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision import models


class Discriminator(nn.Module):
    def __init__(self, style_size):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(style_size + 3, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(512, 1, 4, 1, 0)),
            )
        ])

    def forward(self, x, a):
        x = torch.cat((x, a), dim=1)
        y = [x]
        for layer in self.layers:
            y.append(layer(y[-1]))
        return y[1:]


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, normalize_input=True):
        assert x.dim() == 4 and x.size(1) == 3, 'Wrong input size {}. Should be (N, 3, H, W)'.format(
            tuple(x.size()))
        if normalize_input:
            # Normalize inputs
            # from (-1., 1.), i.e., ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # to ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            x = x + 1 / 2
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)
            x = x.sub(mean[None, :, None, None]).div(std[None, :, None, None])
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out