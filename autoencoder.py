import torch
import torch.nn as nn


class VAutoencoder(nn.Module):
    def initial_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, int(out_channels / 2), 3, padding=1),
                             nn.BatchNorm2d(int(out_channels / 2)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(int(out_channels / 2), out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def consecutive_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels, out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def consecutive_conv_up(self, in_channels, out_channels):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1),  # HEEEERE IT WASS IN OUT
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True),
                             nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),  # HAND HERE IN IN
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def __init__(self, num_channels, num_latents, num_class):
        super(VAutoencoder, self).__init__()

        print('num_channel', num_channels)
        print('num_latent', num_latents)
        self.num_channels = num_channels
        self.conv_initial = self.initial_conv(1, num_channels)
        self.conv_final = nn.Conv2d(num_channels, 1, 3, padding=1)

        self.conv_rest_x_256 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest_x_128 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest_x_64 = self.consecutive_conv(num_channels * 4, num_channels * 8)
        self.conv_rest_x_32 = self.consecutive_conv(num_channels * 8, num_channels * 16)
        self.conv_rest_x_16 = self.consecutive_conv(num_channels * 16, num_channels * 32)
        self.conv_rest_x_8 = self.consecutive_conv(num_channels * 32, num_channels * 64)

        self.conv_rest_u_16 = self.consecutive_conv_up(num_channels * 64, num_channels * 32)
        self.conv_rest_u_32 = self.consecutive_conv_up(num_channels * 32, num_channels * 16)
        self.conv_rest_u_64 = self.consecutive_conv_up(num_channels * 16, num_channels * 8)
        self.conv_rest_u_128 = self.consecutive_conv_up(num_channels * 8, num_channels * 4)
        self.conv_rest_u_256 = self.consecutive_conv_up(num_channels * 4, num_channels * 2)
        self.conv_rest_u_512 = self.consecutive_conv_up(num_channels * 2, num_channels)

        self.contract = nn.MaxPool2d(2, stride=2)
        self.expand = nn.Upsample(scale_factor=2)
        self.linear_enc = nn.Linear(8 * 8 * num_channels * 64, num_latents)
        self.linear_dec = nn.Linear(num_latents, 8 * 8 * num_channels * 64)
        self.classifier_fc1 = nn.Linear(num_latents, num_latents)
        self.classifier = nn.Linear(num_latents, num_class)

    def encoder(self, x):
        x_512 = self.conv_initial(x)  # conv_initial 1->64->128
        x_256 = self.contract(x_512)
        x_256 = self.conv_rest_x_256(x_256)  # rest 128->128->256
        x_128 = self.contract(x_256)
        x_128 = self.conv_rest_x_128(x_128)  # rest 256->256->512
        x_64 = self.contract(x_128)
        x_64 = self.conv_rest_x_64(x_64)  # rest 512->512->256
        x_32 = self.contract(x_64)
        x_32 = self.conv_rest_x_32(x_32)
        x_16 = self.contract(x_32)
        x_16 = self.conv_rest_x_16(x_16)
        x_8 = self.contract(x_16)
        x_8 = self.conv_rest_x_8(x_8)
        x_flat = x_8.view(-1,
                           8 * 8 * self.num_channels * 64)  # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        mean = self.linear_enc(x_flat)
        std = 1.e-6 + nn.functional.softplus(self.linear_enc(x_flat))
        return mean, std

    def decoder(self, x):
        u_8 = self.linear_dec(x).view(-1, self.num_channels * 64, 8, 8)
        u_16 = self.expand(u_8)
        u_16 = self.conv_rest_u_16(u_16)
        u_32 = self.expand(u_16)
        u_32 = self.conv_rest_u_32(u_32)
        u_64 = self.expand(u_32)
        u_64 = self.conv_rest_u_64(u_64)
        u_128 = self.expand(u_64)
        u_128 = self.conv_rest_u_128(u_128)  # rest 256+512-> 512 -> 512
        u_256 = self.expand(u_128)
        u_256 = self.conv_rest_u_256(u_256)  # rest 512+256-> 256 -> 256
        u_512 = self.expand(u_256)
        u_512 = self.conv_rest_u_512(u_512)  # rest 256+128-> 128 -> 128
        u_512 = self.conv_final(u_512)

        S = torch.sigmoid(u_512)
        return S

    def classify(self, x):
        mean, logvar = self.encoder(x) #

        std = torch.exp(
            0.5 * logvar)  # note that the output is log(var), so we need to exp it and take the sqrt in order to get the std
        z = mean + std * torch.randn_like(
            std)  # the z are such so to have the same sampling as N(mean,std) but without messing with the stochastic gradient descent

        fc1 = nn.functional.relu(self.classifier_fc1(z))
        classification = self.classifier(fc1)
        return classification

    def forward(self, x):
        mean, logvar = self.encoder(x) #

        std = torch.exp(
            0.5 * logvar)  # note that the output is log(var), so we need to exp it and take the sqrt in order to get the std
        z = mean + std * torch.randn_like(
            std)  # the z are such so to have the same sampling as N(mean,std) but without messing with the stochastic gradient descent

        fc1 = nn.functional.relu(self.classifier_fc1(z))
        classification = self.classifier(fc1)
        return mean, std, classification, self.decoder(z)  # note that is the logvar that gets returned logvar,

class VAutoencoderDiff(nn.Module):
    def initial_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, int(out_channels / 2), 3, padding=1),
                             nn.BatchNorm2d(int(out_channels / 2)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(int(out_channels / 2), out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def consecutive_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels, out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def consecutive_conv_up(self, in_channels, out_channels):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1),  # HEEEERE IT WASS IN OUT
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True),
                             nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),  # HAND HERE IN IN
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def __init__(self, num_channels, num_latents, num_class):
        super(VAutoencoderDiff, self).__init__()

        print('num_channel', num_channels)
        print('num_latent', num_latents)
        self.num_channels = num_channels
        self.conv_initial = self.initial_conv(1, num_channels)
        self.conv_final = nn.Conv2d(num_channels, 1, 3, padding=1)

        self.conv_rest_x_256 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest_x_128 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest_x_64 = self.consecutive_conv(num_channels * 4, num_channels * 8)
        self.conv_rest_x_32 = self.consecutive_conv(num_channels * 8, num_channels * 16)
        self.conv_rest_x_16 = self.consecutive_conv(num_channels * 16, num_channels * 32)
        self.conv_rest_x_8 = self.consecutive_conv(num_channels * 32, num_channels * 64)

        self.conv_rest_u_16 = self.consecutive_conv_up(num_channels * 64, num_channels * 32)
        self.conv_rest_u_32 = self.consecutive_conv_up(num_channels * 32, num_channels * 16)
        self.conv_rest_u_64 = self.consecutive_conv_up(num_channels * 16, num_channels * 8)
        self.conv_rest_u_128 = self.consecutive_conv_up(num_channels * 8, num_channels * 4)
        self.conv_rest_u_256 = self.consecutive_conv_up(num_channels * 4, num_channels * 2)
        self.conv_rest_u_512 = self.consecutive_conv_up(num_channels * 2, num_channels)

        self.contract = nn.MaxPool2d(2, stride=2)
        self.expand = nn.Upsample(scale_factor=2)
        self.linear_enc = nn.Linear(8 * 8 * num_channels * 64, num_latents)
        self.linear_dec = nn.Linear(num_latents, 8 * 8 * num_channels * 64)
        self.diff_classifier_fc1 = nn.Linear(num_latents, num_latents)
        self.diff_classifier = nn.Linear(num_latents, num_class)

    def encoder(self, x):
        x_512 = self.conv_initial(x)  # conv_initial 1->64->128
        x_256 = self.contract(x_512)
        x_256 = self.conv_rest_x_256(x_256)  # rest 128->128->256
        x_128 = self.contract(x_256)
        x_128 = self.conv_rest_x_128(x_128)  # rest 256->256->512
        x_64 = self.contract(x_128)
        x_64 = self.conv_rest_x_64(x_64)  # rest 512->512->256
        x_32 = self.contract(x_64)
        x_32 = self.conv_rest_x_32(x_32)
        x_16 = self.contract(x_32)
        x_16 = self.conv_rest_x_16(x_16)
        x_8 = self.contract(x_16)
        x_8 = self.conv_rest_x_8(x_8)
        x_flat = x_8.view(-1,
                           8 * 8 * self.num_channels * 64)  # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        mean = self.linear_enc(x_flat)
        std = 1.e-6 + nn.functional.softplus(self.linear_enc(x_flat))
        return mean, std

    def decoder(self, x):
        u_8 = self.linear_dec(x).view(-1, self.num_channels * 64, 8, 8)
        u_16 = self.expand(u_8)
        u_16 = self.conv_rest_u_16(u_16)
        u_32 = self.expand(u_16)
        u_32 = self.conv_rest_u_32(u_32)
        u_64 = self.expand(u_32)
        u_64 = self.conv_rest_u_64(u_64)
        u_128 = self.expand(u_64)
        u_128 = self.conv_rest_u_128(u_128)  # rest 256+512-> 512 -> 512
        u_256 = self.expand(u_128)
        u_256 = self.conv_rest_u_256(u_256)  # rest 512+256-> 256 -> 256
        u_512 = self.expand(u_256)
        u_512 = self.conv_rest_u_512(u_512)  # rest 256+128-> 128 -> 128
        u_512 = self.conv_final(u_512)

        S = torch.sigmoid(u_512)
        return S

    def classify(self, x1, x2):
        mean1, logvar1 = self.encoder(x1)  #
        mean2, logvar2 = self.encoder(x2)  #

        std1 = torch.exp(
            0.5 * logvar1)  # note that the output is log(var), so we need to exp it and take the sqrt in order to get the std
        std2 = torch.exp(
            0.5 * logvar2)  # note that the output is log(var), so we need to exp it and take the sqrt in order to get the std

        eps = torch.randn_like(
            std1)
        z1 = mean1 + std1 * eps  # the z are such so to have the same sampling as N(mean,std) but without messing with the stochastic gradient descent

        z2 = mean2 + std2 * eps  # the z are such so to have the same sampling as N(mean,std) but without messing with the stochastic gradient descent

        fc1 = nn.functional.relu(self.diff_classifier_fc1(z1 - z2))
        classification = self.diff_classifier(fc1)
        return classification

    def forward(self, x1, x2):
        mean1, logvar1 = self.encoder(x1) #
        mean2, logvar2 = self.encoder(x2)  #

        std1 = torch.exp(
            0.5 * logvar1)  # note that the output is log(var), so we need to exp it and take the sqrt in order to get the std
        std2 = torch.exp(
            0.5 * logvar2)  # note that the output is log(var), so we need to exp it and take the sqrt in order to get the std

        eps = torch.randn_like(
            std1)
        z1 = mean1 + std1 * eps # the z are such so to have the same sampling as N(mean,std) but without messing with the stochastic gradient descent

        z2 = mean2 + std2 * eps  # the z are such so to have the same sampling as N(mean,std) but without messing with the stochastic gradient descent

        fc1 = nn.functional.relu(self.diff_classifier_fc1(z1 - z2))
        classification = self.diff_classifier(fc1)
        return mean1, std1, mean2, mean2, classification, self.decoder(z1), self.decoder(z2)  # note that is the logvar that gets returned logvar,

class CNN(nn.Module):
    def initial_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, int(out_channels / 2), 3, padding=1),
                             nn.BatchNorm2d(int(out_channels / 2)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(int(out_channels / 2), out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def consecutive_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels, out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def __init__(self, num_channels, num_latents, num_class):
        super(CNN, self).__init__()

        print('num_channel', num_channels)
        print('num_latent', num_latents)
        self.num_channels = num_channels
        self.conv_initial = self.initial_conv(1, num_channels)
        self.conv_final = nn.Conv2d(num_channels, 1, 3, padding=1)

        self.conv_rest_x_256 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest_x_128 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest_x_64 = self.consecutive_conv(num_channels * 4, num_channels * 8)
        self.conv_rest_x_32 = self.consecutive_conv(num_channels * 8, num_channels * 16)
        self.conv_rest_x_16 = self.consecutive_conv(num_channels * 16, num_channels * 32)
        self.conv_rest_x_8 = self.consecutive_conv(num_channels * 32, num_channels * 64)

        self.contract = nn.MaxPool2d(2, stride=2)
        self.expand = nn.Upsample(scale_factor=2)
        self.linear_enc = nn.Linear(8 * 8 * num_channels * 64, num_latents)
        self.linear_dec = nn.Linear(num_latents, 8 * 8 * num_channels * 64)
        self.classifier_fc1 = nn.Linear(num_latents, num_latents)
        self.classifier = nn.Linear(num_latents, num_class)

    def encoder(self, x):
        x_512 = self.conv_initial(x)  # conv_initial 1->64->128
        x_256 = self.contract(x_512)
        x_256 = self.conv_rest_x_256(x_256)  # rest 128->128->256
        x_128 = self.contract(x_256)
        x_128 = self.conv_rest_x_128(x_128)  # rest 256->256->512
        x_64 = self.contract(x_128)
        x_64 = self.conv_rest_x_64(x_64)  # rest 512->512->256
        x_32 = self.contract(x_64)
        x_32 = self.conv_rest_x_32(x_32)
        x_16 = self.contract(x_32)
        x_16 = self.conv_rest_x_16(x_16)
        x_8 = self.contract(x_16)
        x_8 = self.conv_rest_x_8(x_8)
        x_flat = x_8.view(-1,
                           8 * 8 * self.num_channels * 64)  # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        mean = self.linear_enc(x_flat)
        std = 1.e-6 + nn.functional.softplus(self.linear_enc(x_flat))
        return mean, std

    def forward(self, x):
        x_512 = self.conv_initial(x)  # conv_initial 1->64->128
        x_256 = self.contract(x_512)
        x_256 = self.conv_rest_x_256(x_256)  # rest 128->128->256
        x_128 = self.contract(x_256)
        x_128 = self.conv_rest_x_128(x_128)  # rest 256->256->512
        x_64 = self.contract(x_128)
        x_64 = self.conv_rest_x_64(x_64)  # rest 512->512->256
        x_32 = self.contract(x_64)
        x_32 = self.conv_rest_x_32(x_32)
        x_16 = self.contract(x_32)
        x_16 = self.conv_rest_x_16(x_16)
        x_8 = self.contract(x_16)
        x_8 = self.conv_rest_x_8(x_8)
        x_flat = x_8.view(-1,
                          8 * 8 * self.num_channels * 64)  # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        mean = self.linear_enc(x_flat)

        fc1 = nn.functional.relu(self.classifier_fc1(mean))
        classification = self.classifier(fc1)
        return classification # note that is the logvar that gets returned logvar,


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Autoencoder(nn.Module):
    def initial_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, int(out_channels / 2), 3, padding=1),
                             nn.BatchNorm2d(int(out_channels / 2)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(int(out_channels / 2), out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def consecutive_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1),
                             nn.BatchNorm2d(in_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels, out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def consecutive_conv_up(self, in_channels, out_channels):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1),  # HEEEERE IT WASS IN OUT
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True),
                             nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),  # HAND HERE IN IN
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))

    def __init__(self, num_channels, num_latents, num_class):
        super(Autoencoder, self).__init__()

        print('num_channel', num_channels)
        print('num_latent', num_latents)
        self.num_channels = num_channels
        self.conv_initial = self.initial_conv(1, num_channels)
        self.conv_final = nn.Conv2d(num_channels, 1, 3, padding=1)

        """
        self.conv_rest_x_384 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest_x_192 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest_x_96 = self.consecutive_conv(num_channels * 4, num_channels * 8)
        self.conv_rest_x_48 = self.consecutive_conv(num_channels * 8, num_channels * 16)
        self.conv_rest_x_24 = self.consecutive_conv(num_channels * 16, num_channels * 32)
        self.conv_rest_x_12 = self.consecutive_conv(num_channels * 32, num_channels * 64)
        self.conv_rest_x_6 = self.consecutive_conv(num_channels * 64, num_channels * 128)
        self.conv_rest_x_3 = self.consecutive_conv(num_channels * 128, num_channels * 256)

        self.conv_rest_u_6 = self.consecutive_conv(num_channels * 256, num_channels * 128)
        self.conv_rest_u_12 = self.consecutive_conv(num_channels * 128, num_channels * 64)
        self.conv_rest_u_24 = self.consecutive_conv_up(num_channels * 64, num_channels * 32)
        self.conv_rest_u_48 = self.consecutive_conv_up(num_channels * 32, num_channels * 16)
        self.conv_rest_u_96 = self.consecutive_conv_up(num_channels * 16, num_channels * 8)
        self.conv_rest_u_192 = self.consecutive_conv_up(num_channels * 8, num_channels * 4)
        self.conv_rest_u_384 = self.consecutive_conv_up(num_channels * 4, num_channels * 2)
        self.conv_rest_u_768 = self.consecutive_conv_up(num_channels * 2, num_channels)
        """

        self.conv_rest_x_384 = self._make_layer(ResidualBlock, num_channels, num_channels * 2)
        self.conv_rest_x_192 = self._make_layer(ResidualBlock, num_channels * 2, num_channels * 4)
        self.conv_rest_x_96 = self._make_layer(ResidualBlock, num_channels * 4, num_channels * 8)
        self.conv_rest_x_48 = self._make_layer(ResidualBlock, num_channels * 8, num_channels * 16)
        self.conv_rest_x_24 = self._make_layer(ResidualBlock, num_channels * 16, num_channels * 32)
        self.conv_rest_x_12 = self._make_layer(ResidualBlock, num_channels * 32, num_channels * 64)
        self.conv_rest_x_6 = self._make_layer(ResidualBlock, num_channels * 64, num_channels * 128)
        self.conv_rest_x_3 = self._make_layer(ResidualBlock, num_channels * 128, num_channels * 256)

        self.conv_rest_u_6 = self._make_layer(ResidualBlock, num_channels * 256, num_channels * 128)
        self.conv_rest_u_12 = self._make_layer(ResidualBlock, num_channels * 128, num_channels * 64)
        self.conv_rest_u_24 = self._make_layer(ResidualBlock, num_channels * 64, num_channels * 32)
        self.conv_rest_u_48 = self._make_layer(ResidualBlock, num_channels * 32, num_channels * 16)
        self.conv_rest_u_96 = self._make_layer(ResidualBlock, num_channels * 16, num_channels * 8)
        self.conv_rest_u_192 = self._make_layer(ResidualBlock, num_channels * 8, num_channels * 4)
        self.conv_rest_u_384 = self._make_layer(ResidualBlock, num_channels * 4, num_channels * 2)
        self.conv_rest_u_768 = self._make_layer(ResidualBlock, num_channels * 2, num_channels)

        self.contract = nn.MaxPool2d(2, stride=2)
        self.expand = nn.Upsample(scale_factor=2)
        self.linear_enc = nn.Linear(3 * 3 * num_channels * 256, num_latents)
        self.linear_dec = nn.Linear(num_latents, 3 * 3 * num_channels * 256)
        self.classifier = nn.Linear(num_latents, num_class)

        self.num_latents = num_latents
        self.num_classes = num_class

    def _make_layer(self, block, inplanes, planes, blocks=1, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def encoder(self, x):
        x_768 = self.conv_initial(x)  # conv_initial 1->64->128
        x_384 = self.contract(x_768)
        #print(x_384.size())
        x_384 = self.conv_rest_x_384(x_384)  # rest 128->128->256
        x_192 = self.contract(x_384)
        #print(x_192.size())
        x_192 = self.conv_rest_x_192(x_192)  # rest 256->256->512
        x_96 = self.contract(x_192)
        #print(x_96.size())
        x_96 = self.conv_rest_x_96(x_96)  # rest 512->512->256
        x_48 = self.contract(x_96)
        #print(x_48.size())
        x_48 = self.conv_rest_x_48(x_48)
        x_24 = self.contract(x_48)
        #print(x_24.size())
        x_24 = self.conv_rest_x_24(x_24)
        x_12 = self.contract(x_24)
        #print(x_12.size())
        x_12 = self.conv_rest_x_12(x_12)
        x_6 = self.contract(x_12)
        x_6 = self.conv_rest_x_6(x_6)
        x_3 = self.contract(x_6)
        x_3 = self.conv_rest_x_3(x_3)
        x_flat = x_3.view(-1,
                           3 * 3 * self.num_channels * 256)  # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        mean = self.linear_enc(x_flat)
        #std = 1.e-6 + nn.functional.softplus(self.linear_enc(x_flat))
        return mean #, std

    def decoder(self, x):
        u_3 = self.linear_dec(x).view(-1, self.num_channels * 256, 3, 3)
        u_6 = self.expand(u_3)
        u_6 = self.conv_rest_u_6(u_6)
        u_12 = self.expand(u_6)
        u_12 = self.conv_rest_u_12(u_12)
        u_24 = self.expand(u_12)
        u_24 = self.conv_rest_u_24(u_24)
        u_48 = self.expand(u_24)
        u_48= self.conv_rest_u_48(u_48)
        u_96 = self.expand(u_48)
        u_96 = self.conv_rest_u_96(u_96)  # rest 256+512-> 512 -> 512
        u_192 = self.expand(u_96)
        u_192 = self.conv_rest_u_192(u_192)  # rest 512+256-> 256 -> 256
        u_384 = self.expand(u_192)
        u_384 = self.conv_rest_u_384(u_384)  # rest 256+128-> 128 -> 128
        u_768 = self.expand(u_384)
        u_768 = self.conv_rest_u_768(u_768)
        u_768 = self.conv_final(u_768)

        S = torch.tanh(u_768)
        return S

    def add_classifier(self, num_class):
        weight = torch.zeros((num_class, self.num_latents), requires_grad=True) + 0.1
        print(weight.size())
        weight[:self.num_classes, :] = self.classifier.weight
        bias = torch.zeros((num_class), requires_grad=True) + 0.1
        bias[:self.num_classes] = self.classifier.bias

        self.classifier = nn.Linear(self.num_latents, num_class)
        self.classifier.weight = torch.nn.Parameter(weight)
        self.classifier.bias = torch.nn.Parameter(bias)

    def classify(self, feature):
        return self.classifier(feature)

    def diff_classify(self, feature1, feature2):
        return self.classifier(feature1 - feature2)

    def forward(self, x):
        mean = nn.functional.softplus(self.encoder(x)) #, logvar
        return self.decoder(mean)  # note that is the logvar that gets returned logvar,

    def recon_class(self, x):
        feature = self.encoder(x)
        mean = nn.functional.softplus(feature) #, logvar
        return self.decoder(mean), self.classifier(feature)  # note that is the logvar that gets returned logvar,


