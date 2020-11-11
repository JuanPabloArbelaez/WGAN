import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt



def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """
    Function for visualizing images: Given a tnsor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    """
    image_tensor = (image_tensor + 1 ) / 2
    print(f'F- show_tensor_images () DIMS of image_tensor: \t{image_tensor.size()}\n')
    image_unflat = image_tensor.detach().cpu()
    print(f'F- show_tensor_images () DIMS of image_unflat: \t{image_unflat.size()}\n')
    image_grid= make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def make_grad_hook():
    """
    Function to keep track of gradients for visualization puposes,
    which fills the grads list when using model.apply(grad_hook)
    """
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook
