import torch
from torch import nn
from torchvision.datasets import MNIST  # Training dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.gen(noise)


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.LeakyReLU(0.2))


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image):
        return self.disc(image)


class GAN(pl.LightningModule):
    def __init__(
        self, generator, discriminator, learning_rate=0.00001, z_dim=64, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gen = generator
        self.disc = discriminator
        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        real, _ = batch
        num_images = len(real)
        real = real.view(num_images, -1)
        if optimizer_idx == 0:
            noise = self.get_noise(num_images)
            xFake = self.gen(noise)
            yFakePredicted = self.disc(xFake)
            yFakeLabel = torch.ones_like(yFakePredicted)
            gen_loss = self.criterion(yFakePredicted, yFakeLabel)
            self.log(
                name="gen_loss",
                value=gen_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )

            return gen_loss
        else:
            noise = self.get_noise(num_images)
            with torch.no_grad():
                xFake = self.gen(noise).detach()

            yFakePredicted = self.disc(xFake)
            yFakeLabel = torch.zeros_like(yFakePredicted)
            lossFake = self.criterion(yFakePredicted, yFakeLabel)

            yRealPredicted = self.disc(real)
            yRealLabel = torch.ones_like(yRealPredicted)
            lossReal = self.criterion(yRealPredicted, yRealLabel)

            disc_loss = (lossFake + lossReal) / 2
            self.log(
                name="disc_loss",
                value=disc_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )
            return disc_loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        pass

    def training_epoch_end(self, outputs) -> None:
        num_images = 25
        size = (1, 28, 28)
        fake_noise = self.get_noise(num_images)
        fake = self.gen(fake_noise)

        fig, ax = plt.subplots(figsize=(15, 15))
        image_unflat = fake.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat[:num_images], nrow=5)
        ax.imshow(image_grid.permute(1, 2, 0).squeeze())
        ax.axis("off")
        fig.tight_layout(pad=0)
        # To remove the huge white borders
        ax.margins(0)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        self.logger.experiment.log_image(
            image_from_plot, name="generated_images", overwrite=True
        )

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learning_rate)
        dis_opt = torch.optim.Adam(self.disc.parameters(), lr=self.learning_rate)
        return gen_opt, dis_opt

    def get_noise(self, n_samples):
        return torch.randn(n_samples, self.z_dim, device=self.device)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="..", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage):
        self.dataset = MNIST(
            self.data_dir, download=False, transform=transforms.ToTensor()
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
