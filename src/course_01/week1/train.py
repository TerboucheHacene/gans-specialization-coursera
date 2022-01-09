# import comet_ml at the top of your file
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import torch
import pytorch_lightning as pl

torch.manual_seed(0)  # Set for testing purposes, please do not change!
from nn_utils import GAN, Generator, Discriminator, MNISTDataModule

# Set your parameters
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001


if __name__ == "__main__":
    experiment = CometLogger(
        api_key="F8z2rvZxchPyTT2l1IawCAE7G",
        project_name="gans-specialization",
        workspace="ihssen",
    )

    gen = Generator(z_dim)
    disc = Discriminator()
    gan = GAN(generator=gen, discriminator=disc, learning_rate=lr)

    data_module = MNISTDataModule(batch_size=batch_size)

    trainer = pl.Trainer(max_epochs=n_epochs, logger=experiment)
    trainer.fit(gan, data_module)
