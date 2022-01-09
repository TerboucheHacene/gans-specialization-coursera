import torch
from torch import nn
from torchvision.datasets import MNIST  # Training dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(
        self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False
    ):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def unsqueeze_noise(self, noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(
        self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False
    ):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2),
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


class GAN(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        learning_rate=0.00001,
        z_dim=64,
        beta_1=0.5,
        beta_2=0.999,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gen = generator
        self.disc = discriminator
        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.criterion = nn.BCEWithLogitsLoss()
        self.gen = self.gen.apply(self.weights_init)
        self.disc = self.disc.apply(self.weights_init)

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        real, _ = batch
        num_images = len(real)
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
        fake = (fake + 1) / 2

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
        gen_opt = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
        )
        disc_opt = torch.optim.Adam(
            self.disc.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
        )
        return gen_opt, disc_opt

    def get_noise(self, n_samples):
        return torch.randn(n_samples, self.z_dim, device=self.device)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="..", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.dataset = MNIST(self.data_dir, download=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
