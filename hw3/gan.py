import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from math import prod


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        self.in_size = in_size
        # Size of z latent vector (i.e. size of generator input)
        nz = in_size[0]
        # Size of feature maps in generator
        # Size of feature maps in discriminator
        ndf = 64
        modules = []
        modules += [
            nn.Conv2d(in_size[0], 32, kernel_size=5),
            # nn.Dropout2d(p=0.1),
            nn.ReLU(),

            nn.Conv2d(32, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(p=0.1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.),
            nn.Tanh(),
            nn.Conv2d(256, 256, kernel_size=5, padding=6, stride=2),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(p=0.1),
            nn.Tanh(),

            nn.Conv2d(256, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(p=0.1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=5, padding=7, stride=2),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(p=0.1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 1, kernel_size=8),
            # nn.Dropout2d(p=0.1),
            nn.LeakyReLU(0.1),
        ]
        self.cnn = nn.Sequential(*modules)

        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        y = self.cnn(x).reshape((x.shape[0], 1))
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []
        self.featuremap_size = featuremap_size
        in_c = int(z_dim / (featuremap_size ** 2))
        # Number of channels in the training images. For color images this is 3
        nc = out_channels
        # Size of z latent vector (i.e. size of generator input)
        nz = z_dim
        # Size of feature maps in generator
        ngf = 64

        # Size of feature maps in discriminator
        ndf = 64
        modules = [
            nn.Upsample(scale_factor=2, mode="bicubic"),

            nn.Conv2d(in_c, ngf * 2, kernel_size=5, padding=2),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.1),

            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2, mode="bicubic"),

            nn.Conv2d(ngf * 2, ngf, kernel_size=5, padding=2, dilation=1, stride=1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.1),

            nn.Conv2d(ngf, ngf, kernel_size=5, padding=2, dilation=1, stride=1),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2, mode="bicubic"),

            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(0.1),

            nn.Upsample(scale_factor=2, mode="bicubic"),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, dilation=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ngf, nc, kernel_size=5, padding=2, dilation=1),
            nn.Tanh()

        ]
        #         modules =[
        #             nn.ConvTranspose2d( in_c, ngf * 2, 4, 2, 0, bias=False),
        #             nn.BatchNorm2d(ngf * 2),
        #             nn.LeakyReLU(negative_slope=0.2),

        #             # state size. (ngf*4) x 8 x 8
        #             nn.ConvTranspose2d( ngf*2 , ngf, 4, 2, 1, bias=False),
        #             nn.BatchNorm2d(ngf),
        #             nn.LeakyReLU(negative_slope=0.2),

        #             # state size. (ngf*2) x 16 x 16
        #             nn.ConvTranspose2d( ngf, ngf, 6, 2, 1, bias=False),
        #             nn.BatchNorm2d(ngf),
        #             nn.LeakyReLU(negative_slope=0.2),

        #             # state size. (ngf) x 32 x 32
        #             nn.ConvTranspose2d( ngf, ngf, 5, 3, 1, bias=False),
        #             nn.BatchNorm2d(ngf),
        #             nn.LeakyReLU(negative_slope=0.2),
        #             nn.Conv2d(ngf,int(ngf/2),kernel_size=5,stride=2,padding=2),
        #             nn.BatchNorm2d(int(ngf/2)),
        #             nn.LeakyReLU(negative_slope=0.2),
        #             nn.Conv2d(int(ngf/2),nc,kernel_size=2,stride=1,padding=1),
        #             nn.Tanh()

        #         ]
        #         modules += [
        #             nn.ConvTranspose2d(in_channels=in_c, out_channels=256, kernel_size=5, padding=1, dilation=1,
        #                                stride=2),
        # #             nn.Dropout2d(p=0.3),
        #             nn.BatchNorm2d(num_features=256),
        #             nn.LeakyReLU(negative_slope=0.2),
        #             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, padding=0, dilation=1,
        #                                stride=3),
        # #             nn.Dropout2d(p=0.3),
        #             nn.BatchNorm2d(num_features=128),
        #             nn.LeakyReLU(negative_slope=0.2),
        #             nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=5, padding=0, dilation=1,
        #                                stride=2),
        # #             nn.Dropout2d(p=0.3),
        #             nn.BatchNorm2d(num_features=128),
        #             nn.LeakyReLU(negative_slope=0.2),
        #             nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3,stride=2),
        #             nn.BatchNorm2d(num_features=64),
        #             nn.LeakyReLU(negative_slope=0.2),
        #             nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, padding=2,stride=2, dilation=1),
        #             nn.Tanh()
        #         ]
        self.cnn = nn.Sequential(*modules)

        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        sample = torch.randn(n, self.z_dim).to(device=device) * 10

        if not with_grad:
            with torch.no_grad():
                samples = self.forward(sample)
        else:
            samples = self.forward(sample)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.reshape(z.shape[0], -1, self.featuremap_size, self.featuremap_size)
        x = self.cnn(z)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    data_label_targets = torch.ones_like(y_data) * data_label
    noise_mat = torch.rand_like(y_data) * label_noise - label_noise / 2
    loss_data = loss_fn(y_data, data_label_targets + noise_mat)

    generated_label_target = torch.ones_like(y_generated) * (1 - data_label)
    noise_mat_generated = torch.rand_like(y_generated) * label_noise - label_noise / 2
    loss_generated = loss_fn(y_generated, (generated_label_target + noise_mat_generated))
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss_data_weights = torch.ones_like(y_generated) * data_label
    loss = loss_fn(y_generated, loss_data_weights)
    # ========================
    return loss


def train_batch(
        dsc_model: Discriminator,
        gen_model: Generator,
        dsc_loss_fn: Callable,
        gen_loss_fn: Callable,
        dsc_optimizer: Optimizer,
        gen_optimizer: Optimizer,
        x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    sample = gen_model.sample(x_data.shape[0], with_grad=False)
    dsc_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    dsc_gen_out = dsc_model(sample)
    dsc_real_out = dsc_model(x_data)
    dsc_loss = dsc_loss_fn(dsc_real_out, dsc_gen_out)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    k = 1
    for i in range(k):
        sample = gen_model.sample(x_data.shape[0], with_grad=True)
        dsc_gen_out = dsc_model(sample)
        dsc_optimizer.zero_grad()
        gen_optimizer.zero_grad()
        gen_loss = gen_loss_fn(dsc_gen_out)
        gen_loss.backward()
        gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if len(gen_losses) % 4 == 0:
        torch.save(gen_model, checkpoint_file)
        saved = True
    # ========================

    return saved

