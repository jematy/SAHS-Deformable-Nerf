#这是完全采用spade架构实现的
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torchvision import models


class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        if self.downsample:
            self.downsample_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.residual_downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )


    def forward(self, x):
        identity = x

        out = self.initial(x)
        if self.downsample:
            identity = self.downsample_layer(identity)
            out = self.residual_downsample(out)
        else:
            out = self.residual(out)

        out += identity
        return out


class SPADELayer(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        self.mlp_shared = nn.Sequential(
            # nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
            spectral_norm(nn.Conv2d(label_nc, 128, kernel_size=3, padding=1)),
            nn.ReLU(inplace=False)  # 是否需要Relu不确定
        )

        self.conv_gamma = spectral_norm(nn.Conv2d(128, norm_nc, kernel_size=3, padding=1))
        self.conv_beta = spectral_norm(nn.Conv2d(128, norm_nc, kernel_size=3, padding=1))
        # 是否要spectral_norm

    def forward(self, x, F_id):
        normalized = self.param_free_norm(x)

        F_id = F.interpolate(F_id, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(F_id)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out



class IdEncoder(nn.Module):
    def __init__(self):
        super(IdEncoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.AvgPool2d(2, stride=2)
        )

        self.layer2 = ResBlock2d(64, 64)
        self.layer3 = ResBlock2d(64, 128, downsample=True)
        self.layer4 = ResBlock2d(128, 256, downsample=True)

    def forward(self, x):
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        return x1, x2, x3



class SPADEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fid_channels, downsample=False, upsample=False):
        super(SPADEBlock, self).__init__()
        self.spade1 = SPADELayer(in_channels, fid_channels)
        # self.lrelu1 = nn.LeakyReLU(0.2)
        self.lrelu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1_sn = spectral_norm(self.conv1)
        self.spade2 = SPADELayer(out_channels, fid_channels)
        # self.lrelu2 = nn.LeakyReLU(0.2)
        self.lrelu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)#out_channels 可以解决layer4单layer3就挂了
        self.conv2_sn = spectral_norm(self.conv2)

        # self.downsample = downsample

        # if downsample:
        #     self.downsampler = nn.AvgPool2d(2, stride=2)
        #     self.residual_downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        # self.upsample = upsample
        # if upsample:
        #     self.upsampler = nn.Upsample(scale_factor=2, mode='nearest')
        #     self.residual_upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,
        #                                                 output_padding=1)

        self.spade_s = SPADELayer(in_channels, fid_channels)
        self.lrelu_s = nn.LeakyReLU(0.2)
        self.conv_s = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x, fid):
        identity = x

        x1 = self.spade1(x, fid)
        x1 = self.lrelu1(x1)
        x1 = self.conv1_sn(x1)

        # if self.downsample:
        #     x1 = self.downsampler(x1)
        #     identity = self.residual_downsample(identity)
        # if self.upsample:
        #     x1 = self.upsampler(x1)
        #     identity = self.residual_upsample(identity)

        x2 = self.spade2(x1, fid)
        x2 = self.lrelu2(x2)
        x2 = self.conv2_sn(x2)

        x_ = self.conv_s(self.lrelu_s(self.spade_s(identity, fid)))
        out = x_ + x2
        return out



# class RefineNetwork(nn.Module):
#     def __init__(self, input_channels, fid_channels1, fid_channels2, fid_channels3):
#         super(RefineNetwork, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.AvgPool2d(2, stride=2)
#         )
#
#         self.layer2 = SPADEBlock(64, 64, fid_channels1, downsample=True)
#         self.layer3 = SPADEBlock(64, 128, fid_channels2, downsample=True)
#         # self.layer4 = SPADEBlock(128, 256, fid_channels3)
#         # may it need to upsaml
#         self.layer4 = SPADEBlock(128, 256, fid_channels3, upsample=True)
#         self.layer5 = SPADEBlock(256, 256, fid_channels3, upsample=True)
#         self.layer6 = SPADEBlock(256, 128, fid_channels2, upsample=True)
#         self.layer7 = SPADEBlock(128, 64, fid_channels1, upsample=True)
#         self.layer8 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
#
#     def forward(self, x, fid1, fid2, fid3):
#         x = self.layer1(x)
#         x = self.layer2(x, fid1)
#         x = self.layer3(x, fid2)
#         x = self.layer4(x, fid3)
#         x = self.layer5(x, fid3)
#         x = self.layer6(x, fid2)
#         x = self.layer7(x, fid1)
#         x = self.layer8(x)
#         return x


class Generator(nn.Module):
    def __init__(self, fid_chanels):
        super(Generator, self).__init__()

        self.fc = spectral_norm(nn.Linear(256, 16384))
        self.spade_blocks = nn.ModuleList([
            SPADEBlock(1024, 1024, fid_chanels),
            SPADEBlock(1024, 1024, fid_chanels),
            SPADEBlock(1024, 512, fid_chanels),
            SPADEBlock(512, 256, fid_chanels),
            SPADEBlock(256, 128, fid_chanels),
            SPADEBlock(128, 64, fid_chanels),
        ])
        self.upsample2x = lambda x: F.interpolate(x, scale_factor=2.0, mode='nearest')
        self.conv = nn.sequential(
            spectral_norm(nn.Conv2d(64, 3, kernel_size=3, padding=1)),
            nn.Tanh()
        )
    def forward(self, I_src, I_raw):
        #src is texture
        h = self.fc(I_raw)
        h = h.view(-1, 1024, 4, 4)
        for block in self.spade_blocks:
            h = block(h, I_src)
            h = self.upsample2x(h)
        y = self.conv(h)
        return y

class Discriminator(nn.Module):
    def __init__(self, style_size):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(style_size + 3, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=False),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=False),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=False),
            ),
            nn.Sequential(
                spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=False),
            ),
            nn.Sequential(
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2, inplace=False),
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
        assert x.dim() == 4 and x.size(1) == 3, 'Wrong input size {}. Should be (N, 3, H, W)'.format(tuple(x.size()))
        if normalize_input:
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
