import torch
from torch import nn
from torchvision.datasets import MNIST  # Training dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.nn.functional as F


import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    """
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(
        self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False
    ):
        """
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, input_dim)
        """
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)


class Discriminator(nn.Module):
    """
    Discriminator Class
    Values:
      im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (MNIST is black-and-white, so 1 channel is your default)
      hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(
        self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False
    ):
        """
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        """
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


def combine_vectors(x, y):
    """
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector.
        In this assignment, this will be the noise vector of shape (n_samples, z_dim),
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    """
    combined = torch.cat((x.float(), y.float()), dim=1)
    return combined


class GAN(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminator,
        learning_rate=0.00001,
        z_dim=64,
        n_classes=10,
        input_shape=(1, 28, 28),
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gen = generator
        self.disc = discriminator
        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.criterion = nn.BCEWithLogitsLoss()
        self.gen = self.gen.apply(self.weights_init)
        self.disc = self.disc.apply(self.weights_init)

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        real, labels = batch
        num_images = len(real)

        # Generator
        if optimizer_idx == 0:
            (one_hot_labels, image_one_hot_labels) = self.get_one_hot_labales_and_images(
                labels
            )
            noise = self.get_noise(num_images)
            noise_and_labels = combine_vectors(noise, one_hot_labels)
            fake = self.gen(noise_and_labels)
            fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)

            disc_fake_pred = self.disc(fake_image_and_labels)
            gen_loss = self.criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            self.log(
                name="gen_loss",
                value=gen_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )

            return gen_loss
        # Discriminator
        if optimizer_idx == 1:

            (one_hot_labels, image_one_hot_labels) = self.get_one_hot_labales_and_images(
                labels
            )
            noise = self.get_noise(num_images)
            noise_and_labels = combine_vectors(noise, one_hot_labels)
            with torch.no_grad():
                fake = self.gen(noise_and_labels).detach()

            fake_image_and_labels = combine_vectors(fake.detach(), image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
            disc_fake_pred = self.disc(fake_image_and_labels)
            disc_real_pred = self.disc(real_image_and_labels)

            disc_fake_loss = self.criterion(
                disc_fake_pred, torch.zeros_like(disc_fake_pred)
            )
            disc_real_loss = self.criterion(
                disc_real_pred, torch.ones_like(disc_real_pred)
            )
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
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
            image_from_plot,
            name="generated_images",
            overwrite=False,
        )

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.learning_rate)
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.learning_rate)
        return [gen_opt, disc_opt], []

    def get_noise(self, n_samples):
        return torch.randn(n_samples, self.z_dim, device=self.device)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def get_one_hot_labales_and_images(self, labels):
        one_hot_labels = F.one_hot(labels, self.n_classes)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(
            1, 1, self.input_shape[1], self.input_shape[2]
        )
        return (one_hot_labels, image_one_hot_labels)


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
