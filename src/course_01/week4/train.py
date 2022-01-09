# import comet_ml at the top of your file
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import torch
import pytorch_lightning as pl

torch.manual_seed(0)  # Set for testing purposes, please do not change!
from nn_utils import GAN, Discriminator, Generator, Discriminator, MNISTDataModule


def get_input_dimensions(z_dim, mnist_shape, n_classes):
    """
    Function for getting the size of the conditional input dimensions
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns:
        generator_input_dim: the input dimensionality of the conditional generator,
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    """
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = n_classes + mnist_shape[0]
    return generator_input_dim, discriminator_im_chan


mnist_shape = (1, 28, 28)
n_classes = 10
n_epochs = 200
z_dim = 64
batch_size = 128
lr = 0.0002


if __name__ == "__main__":
    experiment = CometLogger(
        api_key="F8z2rvZxchPyTT2l1IawCAE7G",
        project_name="gans-specialization",
        workspace="ihssen",
    )

    generator_input_dim, discriminator_im_chan = get_input_dimensions(
        z_dim, mnist_shape, n_classes
    )

    gen = Generator(input_dim=generator_input_dim)
    disc = Discriminator(im_chan=discriminator_im_chan)
    gan = GAN(
        generator=gen,
        discriminator=disc,
        z_dim=z_dim,
        n_classes=n_classes,
        input_shape=mnist_shape,
        learning_rate=lr,
    )

    data_module = MNISTDataModule(batch_size=batch_size)

    trainer = pl.Trainer(max_epochs=n_epochs, logger=experiment)
    trainer.fit(gan, data_module)
