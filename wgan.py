import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader



class Generator(nn.Module):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (MNIST is black-and-white, so 1 channel)
        hidden_dim: the inner dimension, a scalar
    """
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build Neural Network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim,*4),
            self.make_gen_block(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim*2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernle_size=4, final_layer=True),            
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        """
        Function to return a seuquence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: size of convolutional filter (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, True if it is the final layer and false otherwise
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
                nn.Tanh()
            )

    def forward(self, noise):
        """
        Method for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise vector of with dimensions (n_samples, z_dim)
        """
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)


def get_noise(n_samples, z_dim, device='cpu'):
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
            (MNIST is blacn-and-white, so 1 channel)
        hidden_dim: the inner dimension, a scalar
    """
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim*2),
            self.make_crit_block(hidden_dim*2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        """
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm(except in the final layer), and an activation (sexcept in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_siz: size of convolutional filter (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a bool, True if it is the final layer and False otherwise
                (affects activation)
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
        Function for completing a forward apps of the critic: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        """
        print(f'The dimensions of the image to the crit is: \t{image.size()}\n')
        crit_pred = self.crit(image)
        print(f'The dimensions of the crit forward() is: \t{crit_pred.view(len(crit_pred), -1)}\n')
        return crit_pred.view(len(crit_pred), -1)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.wieght, 0.0, 0.2)
        torch.nn.init.normal_(m.bias, 0)


def get_gradient(crit, real, fake, epsilon):
    """
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of uniformly random propportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    """
    # Mix the images togeteher
    mixed_images = real*epsilon + fake*(1-epsilon)

    # calculate the critic's scores on the mixed images
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
    return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each of images's gradient
    and peanlize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed_image
    Returns:
        penalty: the gradient penalty
    """
    # Flatten the gradient so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the man squared distance of the gradient norms from 1
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
        c_lambda: the current wieght of the gradient penalty
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    """
    crit_loss = torch.mean(crit_fake_pred - crit_real_pred + (c_lamda * gp))
    return crit_loss