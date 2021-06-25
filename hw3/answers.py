r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 100
    hypers['seq_len'] = 64
    hypers['h_dim'] = 512
    hypers['n_layers'] = 2
    hypers['dropout'] = 0.11
    hypers['learn_rate'] = 1e-2
    hypers['lr_sched_factor'] = 0.1
    hypers['lr_sched_patience'] = 1

    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "The folly of man"
    temperature = 0.51
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

We split the corpus into sequences because we want to be reserved with vram.
Also, learning on the entire text, would give us "too much" context. It will try to find connections between
sentences far apart instead of adjusting the model for better close proximity context.
To achieve both, we will require a deeper model which will increase the risk for exploding gradients and vanishing.   
"""

part1_q2 = r"""
The memory of the model contains a hidden state which saves data from previous batches. Therefore, we have prior knowledge of
previous sentences (and longer memory).
"""

part1_q3 = r"""
We don't shuffle the order of batches because there is a logical order for the batches. We would like the model to learn
those higher level contextual relations in order to generate meaningful sentences. Keeping the order, will preserve the
original meaning of the text, which will in turn, feed our model with more meaningful information (and will not render useless
our hidden state that we worked so hard to preserve).
"""

part1_q4 = r"""
1. When learning, we can risk making mistakes (same as making mistakes in class, we learn from them). When actually
generating, we don't want to risk mistakes and we would rather having a less "rich" result than nonsensical one.
2. For a very high temperature, we would get an approximately uniform result. i.e. the higher the temperature, the less bias
the probability for one word or another and the result will make less sense as the network will choose approximately at random 
from every word from the corpus.
3. For a very low temperature, we will have choose with a rising probability the best fitting word from the corpus.
For an extremely low value of the temperature, we will choose the argmax. This is also bad because it will decrease the variety
and will increase the chance for looping text.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['h_dim'] = 128
    hypers['z_dim'] = 16
    hypers['learn_rate'] = 0.0003
    hypers['betas'] = 0.9, 0.999
    hypers['x_sigma2'] = 0.004
    # ========================
    return hypers


part2_q1 = r"""
The $\sigma^2$ is used for regularization purposes; std equal to one. this regularization is
applied to the data term against the KLDiv loss, therefore, encouraging or discouraging the loss to 
attempt to minimize that term, which is the distance between the original image and its enc-dec.
"""

part2_q2 = r"""
1. Let us consider the formula. the first term, the data term, corresponds to the distance of the decoded-encoding of 
a specific image, to its original encoding. it would be minimized when the image corresponds directly
to the decoding of its encoding. this part could be considered "looking" at the ensemble of 
both encoder and decoder.
the second part, the KL divergence loss, specifically correlates to the distance of the encoded
probability distribution from the encoder, to the normal, gaussian distribution
we assume Z, the latent space is. this term will allow us to sample from the latent space and decode 
"fake" results, thus, creating GWB ourselves.
2. the latent space dist would be trained to be closer to the normal gaussian distribution, therefore, creating a more 
"meaningful" encoded results such that the decoder would get a more meaningful data space it can train on.
3. the benefits, as desecribed in 2, are that our latenst space representation will generate a more meaningful 
space of distribution for our decoder model.
"""

part2_q3 = r"""
As there is no easy way to model similarity of unknown results to our problem domain, we would like to
instead model the probability of a result ot be real. This would entail maximizing the evidence distribution
as we have done. 
"""

part2_q4 = r"""
$\sigma^2$ should be between 0 and 1 and therefore, we would meet the problem of vanishing gradients.
Thus, modelling the log would entail the variable's value be between $-\infty$ and $0$.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=8,
        z_dim=16,
        data_label=1,
        label_noise=0.08,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.001,
            betas=(0.5, 0.999),
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0005,
            betas=(0.5, 0.999),
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    return hypers


part3_q1 = r"""
**Your answer:**
When training the discriminator, we don't want to calculate the gradients on the generator, 
as this, would in turn, change the parameters of the generator in a way that would "hurt" the generator
while training the discriminator. Decreasing the discriminator loss directly correlates to
making the generator worse in this case, which is a big no no.
While training the generator, we would very much love it if the generator would have gradients, as without it,
we are lost.
In short, when training the discriminator, we want to use the generator without modifying the generator model.
"""

part3_q2 = r"""
**Your answer:**
1. During training, both the generator and the discriminator are being trained together.
This entails that the discriminator is also being trained and improved "as we go" which means that the loss
of the generator at a certain point has much less meaning by itself.
2. When the loss of the discriminator remains approximately constant with the loss of the generator
decreasing, it means that the generator learned to create examples that directly cause the discrimnator to always 
provide a false prediction, therefore, leading to no gradients that allow the discriminator to learn, while the generator
is still improving and getting better at fooling the atrophied discriminator. this phenomenon is called 
Gradient Collapse.
"""

part3_q3 = r"""
**Your answer:**
Using GAN, we get much more varying results that look less like existing samples from the data set.
In the VAE, on the other hand, we get better looking results much earlier in the training process.
As the generator learns by fooling the discriminator in the GAN architecture, it must learn more
general features of the shape than the exact replica of the original dataset, and therefore,
will take longer to actually learn a meaningful representation and fine details of the actual feature space
we are training on.
In the VAE model, the training process improves more the quality of the fakes rather than the details.
"""

# ==============





