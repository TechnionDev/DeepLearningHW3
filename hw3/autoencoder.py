import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


class SkipConnection(nn.Module):
    """
    A skip connection module
    """

    def __init__(self,
                 main_path,
                 in_channels,
                 out_channels):
        super().__init__()
        self.main_path = main_path
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels=in_channels,
                                                                                    out_channels=out_channels,
                                                                                    kernel_size=1)

    def forward(self, X):
        return self.main_path(X) + self.shortcut(X)


class InceptionBlock(nn.Module):
    """
    Generate a general purpose Inception block
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_sizes: Sequence[int] = [1, 3, 5],
            batchnorm: bool = False,
            dropout: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        assert (out_channels % 4 == 0)
        out_channels = int(out_channels / 4)
        self.wide_layer = []
        pooling = [
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1)]
        self.wide_layer += [pooling]
        self.wide_layer += [[nn.Conv2d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       padding=int((kernel_size - 1) / 2),
                                       bias=True)] for kernel_size in kernel_sizes]
        if dropout > 0:
            self.wide_layer = [conv + [nn.Dropout2d(p=dropout)] for conv in self.wide_layer]
        if batchnorm:
            self.wide_layer = [conv + [nn.BatchNorm2d(num_features=out_channels)] for conv in self.wide_layer]
        self.wide_layer = [nn.Sequential(*filt) for filt in self.wide_layer]
        self.wide_layer = nn.ModuleList(self.wide_layer)
        self.activation_layer = nn.ELU()

    def forward(self, x):
        output = [conv_filter(x) for conv_filter in self.wide_layer]
        activated = [self.activation_layer(element) for element in output]
        return torch.cat(activated, 1)


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        layers = [64, 128] * 3
        in_c = in_channels
        for i, layer in enumerate(layers):
            modules += [
                SkipConnection(InceptionBlock(in_channels=in_c, out_channels=layer, batchnorm=True), in_channels=in_c,
                               out_channels=layer)]
            in_c = layer
        modules += [nn.Conv2d(in_channels=layers[-1], out_channels=out_channels, kernel_size=1)]

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)[
        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        layers = [64, 128] * 3
        in_c = in_channels
        for i, layer in enumerate(layers):
            modules += [
                SkipConnection(InceptionBlock(in_channels=in_c, out_channels=layer, batchnorm=True), in_channels=in_c,
                               out_channels=layer)]
            in_c = layer
        modules += [nn.Conv2d(in_channels=layers[-1], out_channels=out_channels, kernel_size=1)]

        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.mu_layer = nn.Linear(n_features, z_dim, bias=True)
        self.sigma_layer = nn.Linear(n_features, z_dim, bias=True)
        self.z_decoding_layer = nn.Linear(z_dim, n_features, bias=True)

        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        h = self.features_encoder(x).reshape(x.shape[0], -1)
        sample = torch.randn((h.shape[0], self.z_dim))
        mu = self.mu_layer(h)
        log_sigma2 = self.sigma_layer(h)
        z = mu + sample * log_sigma2
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h = self.z_decoding_layer(z)
        h = h.reshape((h.shape[0], *self.features_shape))
        x_rec = self.features_decoder(h)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            # todo: from where should i get the mean!?
            z = torch.randn((n, self.z_dim), device=device)
            samples = self.decode(z)
            # ========================

        # Detach and move to CPU for display purposes
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    # TODO: RECONSIDER IMPL, MINE DOESNT WORK SARI DOES
    dx = torch.prod(torch.tensor(x.shape[1:]))
    dz = z_mu.shape[1]

    data_loss = (((x - xr).reshape(x.shape[0], -1).norm(dim=-1) ** 2) / (x_sigma2 * dx)).mean()
    var_mat = torch.exp(z_log_sigma2)
    print(var_mat.shape)
    kldiv_loss = torch.exp(z_log_sigma2).sum(dim=-1) - z_log_sigma2.sum(dim=-1) + z_mu.norm(dim=-1) ** 2 - dz
    loss = data_loss + kldiv_loss.mean()
    # ========================

    return loss, data_loss, kldiv_loss
