# import comet_ml at the top of your file
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import torch
import pytorch_lightning as pl

torch.manual_seed(0)  # Set for testing purposes, please do not change!
from nn_utils import GAN, Generator, Discriminator, MNISTDataModule


z_dim = 64
display_step = 500
batch_size = 128
# A learning rate of 0.0002 works well on DCGAN
lr = 0.0002
n_epochs = 50
beta_1 = 0.5
beta_2 = 0.999

if __name__ == "__main__":
    experiment = CometLogger(
        api_key="F8z2rvZxchPyTT2l1IawCAE7G",
        project_name="gans-specialization",
        workspace="ihssen",
    )

    gen = Generator(z_dim)
    disc = Discriminator()
    gan = GAN(
        generator=gen,
        discriminator=disc,
        z_dim=z_dim,
        learning_rate=lr,
        beta_1=beta_1,
        beta_2=beta_2,
    )

    data_module = MNISTDataModule(batch_size=batch_size)

    trainer = pl.Trainer(max_epochs=n_epochs, logger=experiment)
    trainer.fit(gan, data_module)
