import torch
from torch import nn
from torchvision.datasets import MNIST  # Training dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)  # Set for testing purposes, please do not change!


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def make_grad_hook():
    """
    Function to keep track of gradients for visualization purposes,
    which fills the grads list when using model.apply(grad_hook).
    """
    grads = []

    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)

    return grads, grad_hook


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
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


def get_noise(n_samples, z_dim, device="cpu"):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    """
    return torch.randn(n_samples, z_dim, device=device)


class Critic(nn.Module):
    """
    Critic Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(
        self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False
    ):
        """
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
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
        Function for completing a forward pass of the critic: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        """
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)


def get_gradient(crit, real, fake, epsilon):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    """
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    """
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gen_loss(crit_fake_pred):
    """
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    """
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    """
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    """
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    return crit_loss


class GAN(pl.LightningModule):
    def __init__(
        self,
        generator,
        critic,
        z_dim=64,
        learning_rate=0.00001,
        beta_1=0.5,
        beta_2=0.999,
        c_lambda=10,
        crit_repeats=5,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gen = generator
        self.crit = critic
        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.c_lambda = c_lambda
        self.crit_repeats = crit_repeats
        self.gen = self.gen.apply(self.weights_init)
        self.crit = self.crit.apply(self.weights_init)

    def training_step(self, batch, batch_idx, optimizer_idx, *args, **kwargs):
        real, _ = batch
        num_images = len(real)

        ## Generator loss
        if optimizer_idx == 0:
            noise = self.get_noise(num_images)
            xFake = self.gen(noise)
            crit_fake_pred = self.crit(xFake)
            gen_loss = self.get_gen_loss(crit_fake_pred)
            self.log(
                name="gen_loss",
                value=gen_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )

            return gen_loss
        ## Critic loss
        else:
            noise = self.get_noise(num_images)
            with torch.no_grad():
                xFake = self.gen(noise).detach()

            crit_fake_pred = self.crit(xFake)
            crit_real_pred = self.crit(real)

            epsilon = torch.rand(
                len(real), 1, 1, 1, device=self.device, requires_grad=True
            )
            gradient = self.get_gradient(real, xFake.detach(), epsilon)
            gp = self.gradient_penalty(gradient)
            crit_loss = self.get_critic_loss(crit_fake_pred, crit_real_pred, gp)
            self.log(
                name="crit_loss",
                value=crit_loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=False,
            )
            return crit_loss

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
        crit_opt = torch.optim.Adam(
            self.crit.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
        )
        return (
            {"optimizer": gen_opt, "frequency": 1},
            {"optimizer": crit_opt, "frequency": self.crit_repeats},
        )

    def get_noise(self, n_samples):
        return torch.randn(n_samples, self.z_dim, device=self.device)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def get_gradient(self, real, fake, epsilon):
        # Mix the images together
        mixed_images = real * epsilon + fake * (1 - epsilon)
        # Calculate the critic's scores on the mixed images
        mixed_scores = self.crit(mixed_images)
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        return gradient

    def gradient_penalty(self, gradient):
        # Flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)
        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)
        # Penalize the mean squared distance of the gradient norms from 1
        penalty = torch.mean((gradient_norm - 1) ** 2)
        return penalty

    def get_gen_loss(self, crit_fake_pred):
        gen_loss = -torch.mean(crit_fake_pred)
        return gen_loss

    def get_critic_loss(self, crit_fake_pred, crit_real_pred, gp):
        crit_loss = (
            torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + self.c_lambda * gp
        )
        return crit_loss


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
