# Standard library imports
import base64
import io
import os
import random
from math import sqrt

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# PyTorch imports
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image



#mapping_network_path = r"C:\Users\mzlwm\OneDrive\Desktop\MEDGAN\StyleGan2\StyleGAN2-256\StyleGAN2-256\StyleGAN2-256-Meningioma\mapping_net.pth"  # Replace with actual path
# Define the seed function for reproducibility
def seed_everything():
    seed = random.randint(0, 2**32 - 1)  # Generate a random seed between 0 and 2^32-1
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
LOG_RESOLUTION = 8
Z_DIM = 256
W_DIM = 256
LAMBDA_GP = 10
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim)
        )

    def forward(self, x):
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)  # for PixelNorm
        return self.mapping(x)
    
    
class Generator_SG2(nn.Module):

    def __init__(self, log_resolution, W_DIM, n_features = 32, max_features = 256, device="cpu"):

        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(W_DIM, features[0], features[0])
        self.to_rgb = ToRGB(W_DIM, features[0])
        self.device = torch.device(device)

        blocks = [GeneratorBlock(W_DIM, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):

        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        return torch.tanh(rgb)
    

class GeneratorBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)

        self.to_rgb = ToRGB(W_DIM, out_features)

    def forward(self, x, w, noise):

        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb


class StyleBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.to_style = EqualizedLinear(W_DIM, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):

        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])

class ToRGB(nn.Module):

    def __init__(self, W_DIM, features):

        super().__init__()
        self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)

        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):

        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])
    
class Conv2dWeightModulate(nn.Module):

    def __init__(self, in_features, out_features, kernel_size,
                 demodulate = True, eps = 1e-8):

        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):

        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)

class EqualizedLinear(nn.Module):

    def __init__(self, in_features, out_features, bias = 0.):

        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)
    
class EqualizedConv2d(nn.Module):

    def __init__(self, in_features, out_features,
                 kernel_size, padding = 0):

        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)
    
class EqualizedWeight(nn.Module):

    def __init__(self, shape):

        super().__init__()

        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c
    
class PathLengthPenalty(nn.Module):

    def __init__(self, beta):

        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):

        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / sqrt(image_size)
        sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:

            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss
    
def gradient_penalty(critic, real, fake,device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def get_w(batch_size, mapping_net, device=None):
    # Use the provided device (or mapping network device) to avoid cuda/cpu mismatch.
    if device is None:
        device = next(mapping_net.parameters()).device
    z = torch.randn(batch_size, W_DIM, device=device)
    w = mapping_net(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

def get_noise(batch_size, device=None):
        if device is None:
            device = DEVICE

        noise = []
        resolution = 4

        for i in range(LOG_RESOLUTION):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

            noise.append((n1, n2))

            resolution *= 2

        return noise

def generate_examples(gen, epoch, n=100):
    gen.eval()
    alpha = 1.0
    base_dir = '/content/drive/MyDrive/StyleGAN2-256/StyleGAN2-256-Pituitary/saved_examples'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    epoch_dir = os.path.join(base_dir, f'epoch{epoch}')
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    for i in range(n):
        with torch.no_grad():
            w = get_w(1)
            noise = get_noise(1)
            img = gen(w, noise)
            save_image(img * 0.5 + 0.5, os.path.join(epoch_dir, f"img_{i}.png"))


def generate_example_and_show_SG2(gen, mapping_net, steps=1, n=1):
    """
    Generate and display an example image using StyleGAN2.

    Args:
        gen (torch.nn.Module): Generator model.
        mapping_net (torch.nn.Module): Mapping network model.
        steps (int): Number of steps (unused in StyleGAN2 but kept for consistency).
        n (int): Number of examples to generate (unused here but retained for consistency).

    Returns:
        matplotlib.figure.Figure: A matplotlib figure containing the generated image.
    """
    gen.eval()  # Set the generator to evaluation mode
    mapping_net.eval()  # Set the mapping network to evaluation mode

    gen_device = next(gen.parameters()).device
    map_device = next(mapping_net.parameters()).device
    if map_device != gen_device:
        mapping_net = mapping_net.to(gen_device)
        map_device = gen_device

    with torch.no_grad():
        # Generate latent vector and noise
        w = get_w(1, mapping_net, device=map_device)  # Generate a single latent vector
        noise = get_noise(1, device=gen_device)  # Generate noise for all resolution levels

        # Generate the image using the generator
        img = gen(w, noise).to("cpu") * 0.5 + 0.5  # Scale to [0, 1]

        # Prepare the matplotlib figure for display
        fig, ax = plt.subplots(figsize=(2, 2))
        fig.patch.set_facecolor('white')  # Set the figure's background color
        ax.imshow(img[0].permute(1, 2, 0).numpy())  # Convert tensor to HxWxC for plt.imshow
        ax.axis('off')  # Turn off axis
        ax.set_facecolor('white')  # Set the axis background color
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove space around image

        return fig
    

def generate_examples_SG2(generator, mapping_net, num_images):
    """
    Generate images using a StyleGAN2 generator and return them as Base64-encoded strings.

    Args:
        generator (torch.nn.Module): Pre-trained StyleGAN2 generator model.
        mapping_net (torch.nn.Module): Pre-trained mapping network model.
        num_images (int): Number of images to generate.

    Returns:
        list: Base64-encoded images.
        list: BytesIO image buffers for optional ZIP creation.
    """
    images_base64 = []
    image_buffers = []

    generator.eval()  # Set generator to evaluation mode
    mapping_net.eval()  # Set mapping network to evaluation mode
    gen_device = next(generator.parameters()).device
    map_device = next(mapping_net.parameters()).device
    # Keep both modules on the same device for inference.
    if map_device != gen_device:
        mapping_net = mapping_net.to(gen_device)
        map_device = gen_device

    with torch.no_grad():
        for i in range(num_images):
            # Generate latent vector and noise
            w = get_w(1, mapping_net, device=map_device)  # Generate a single latent vector
            noise = get_noise(1, device=gen_device)  # Generate noise for all resolution levels

            # Generate the image using the generator
            img = generator(w, noise).to("cpu") * 0.5 + 0.5  # Normalize to [0, 1]
            img_np = img[0].permute(1, 2, 0).numpy()  # Convert tensor to HxWxC format

            # Convert to Base64-encoded image
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(img_np)
            ax.axis('off')
            buf = io.BytesIO()
            FigureCanvas(fig).print_png(buf)
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            images_base64.append(image_base64)

            # Store the buffer for optional ZIP file creation
            image_buffers.append(buf)
            plt.close(fig)

    return images_base64, image_buffers



