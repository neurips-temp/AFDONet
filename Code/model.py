import torch.nn as nn
import torch.nn.functional as F
import torch

class DynamicCKNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rhos):
        super(DynamicCKNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rhos = rhos

        self.dynamic_weight = nn.Sequential(
            nn.Linear(in_channels, in_channels * out_channels * kernel_size * kernel_size),
            nn.ReLU()
        )

    def forward(self, x, z):
        batch_size = x.size(0)
        dynamic_kernel = self.dynamic_weight(z).view(batch_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        x = x.view(1, batch_size * self.in_channels, *x.shape[2:])
        output = F.conv2d(x, dynamic_kernel, groups=batch_size)
        output = output.view(batch_size, self.out_channels, *output.shape[2:])
        for i in range(len(self.rhos)):
            output[:, i, :, :] *= self.rhos[i]
        return output + 0.5

class Feature_Map(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')
        out_ft = torch.zeros(B, self.out_channels, H, W//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2],
            torch.view_as_complex(self.weights))
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return x    

class Latent_to_RKHS(nn.Module):
    def __init__(self, channels, param_1=12, param_2=12):
        super().__init__()
        self.weights = Feature_Map(channels, channels, param_1, param_2)
        self.conv_kernel = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        x1 = self.weights(x)
        x2 = self.conv_kernel (x)
        return F.gelu(self.norm(x + x1 + x2))

class AFDONet(nn.Module):
    def __init__(self, input_dim, inter_dim, latent_dim, rhos, height=64, width=64, channels=32):
        super(AFDONet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2)
        )
        self.fc = nn.Linear(latent_dim, channels * height * width)
        self.decoder = nn.Sequential(Latent_to_RKHS(channels), Latent_to_RKHS(channels), nn.Conv2d(channels, 2, kernel_size=1))
        self.dynamic_conv = DynamicCKNLayer(2, 2, 3, rhos)

    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        batch = x.size(0)
        x_flat = x.reshape(batch, -1)
        h = self.encoder(x_flat)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterize(mu, logvar)
        input_d = self.fc(z).view(batch, -1, 64, 64)
        recon_x = self.decoder(input_d)
        return recon_x.unsqueeze(1), mu, logvar
    
