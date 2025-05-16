from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, zdim=1000, in_channel=3, img_sz=32, out_channel=128):
        super(Generator, self).__init__()
        self.in_channel = in_channel
        self.img_sz = img_sz
        self.out_channel = out_channel

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(out_channel * (img_sz//4) * (img_sz//4), zdim)
        )

        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(
            nn.Linear(zdim, out_channel * self.init_size**2)
        )

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(out_channel),
        )
        
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(out_channel, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.out_channel, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def sample(self, size, device, detect_data_loader):
        random_indices = torch.randperm(len(detect_data_loader.dataset))[:size]
        random_samples = [detect_data_loader.dataset[i] for i in random_indices]
        detect_data_loader = torch.utils.data.DataLoader(random_samples, batch_size=32)
        for idx, (real_samples, _) in enumerate(detect_data_loader):
            break
        real_samples = real_samples.to(device)
        
        z = self.encode(real_samples)
        
        # noise = torch.randn_like(z) * 0.1
        # z = z + noise

        X = self.decode(z)
        return X 
