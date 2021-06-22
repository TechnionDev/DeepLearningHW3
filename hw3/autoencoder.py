import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod


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
        modules += [
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=5, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout2d(p=0.4),
            nn.ELU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=5, padding=1, dilation=2, stride=2),
            nn.BatchNorm2d(num_features=512),
            nn.Dropout2d(p=0.4),
            nn.ELU(),
            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=5, padding=3, dilation=2),
            nn.Sigmoid()
        ]
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        modules += [
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=512, kernel_size=5, padding=1, dilation=2,
                               stride=2),
            nn.Dropout2d(p=0.1),
            nn.ELU(),
            #             nn.BatchNorm2d(num_features=512),
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=5, padding=0, dilation=1),
            nn.Dropout2d(p=0.1),
            nn.ELU(),
            #             nn.BatchNorm2d(num_features=128),
            nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, padding=0, dilation=1),
            #             nn.ELU(),
            #             nn.Dropout2d(p=0.2),
            #             nn.BatchNorm2d(num_features=out_channels)
        ]
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
        sample = torch.randn((h.shape[0], self.z_dim)).to(device=h.device)
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
    dx = prod(x.shape[1:])
    dz = z_mu.shape[1]
    reshaped_dist = (x - xr).reshape(x.shape[0], -1)
    data_loss = ((torch.norm(reshaped_dist, p=2, dim=-1)) ** 2 / (dx * x_sigma2))
    log_det_var = torch.sum(z_log_sigma2, dim=-1)
    trace_var = torch.sum(torch.exp(z_log_sigma2), dim=-1)
    kldiv_loss = torch.norm(z_mu, 2, -1) ** 2 + trace_var - dz - log_det_var

    data_loss = data_loss.mean()
    kldiv_loss = kldiv_loss.mean()
    loss = (data_loss + kldiv_loss)
    # ========================

    return loss, data_loss, kldiv_loss
