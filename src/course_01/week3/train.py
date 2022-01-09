# import comet_ml at the top of your file
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import torch
import pytorch_lightning as pl

torch.manual_seed(0)  # Set for testing purposes, please do not change!
from nn_utils import GAN, Generator, Critic, MNISTDataModule


n_epochs = 100
z_dim = 64
batch_size = 128
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5

if __name__ == "__main__":
    experiment = CometLogger(
        api_key="F8z2rvZxchPyTT2l1IawCAE7G",
        project_name="gans-specialization",
        workspace="ihssen",
    )

    gen = Generator(z_dim)
    crit = Critic()
    gan = GAN(
        generator=gen,
        critic=crit,
        z_dim=z_dim,
        learning_rate=lr,
        beta_1=beta_1,
        beta_2=beta_2,
        c_lambda=c_lambda,
        crit_repeats=crit_repeats,
    )

    data_module = MNISTDataModule(batch_size=batch_size)

    trainer = pl.Trainer(max_epochs=n_epochs, logger=experiment)
    trainer.fit(gan, data_module)
