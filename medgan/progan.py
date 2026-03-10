# Standard library imports
import base64
import io
import os
import random

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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



START_TRAIN_AT_IMG_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16]
IMAGE_SIZE = 128
CHANNELS_IMG = 3
Z_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

# Define the model components
class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.pn = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x


class Generator_ProGAN(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super(Generator_ProGAN, self).__init__()

        # initial takes 1x1 -> 4x4
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm(),
        )

        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]), 
        )

        for i in range(len(factors) - 1):  # -1 to prevent index error because of factors[i+1]
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


def generate_example_and_show_ProGAN_1(gen, steps, n=1):
    gen.eval()  # Set the model to evaluation mode
    alpha = 1.0
    with torch.no_grad():  # Disable gradient computation
        noise = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)  # Generate random noise
        img = gen(noise, alpha, steps)  # Generate an image
        img = (img * 0.5 + 0.5).clamp(0, 1)  # Normalize the image to [0, 1] range

        # Display the image with enhancements
        fig, ax = plt.subplots(figsize=(2, 2))
        fig.patch.set_facecolor('white')  # Set the figure's background color
        ax.imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy())  # Convert to HxWxC for plt.imshow
        ax.axis('off')  # Turn off axis
        ax.set_facecolor('white')  # Set the axis background color
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove space around image
        return fig
    


def generate_examples_ProGAN(generator, noise, num_images):
    """
    Generate images using a ProGAN generator and return them as Base64-encoded strings.

    Args:
        generator (torch.nn.Module): Pre-trained ProGAN generator model.
        noise (torch.Tensor): Random noise tensor for image generation.
        num_images (int): Number of images to generate.
        steps (int): Number of steps for the ProGAN fade-in mechanism.

    Returns:
        list: Base64-encoded images.
        list: BytesIO image buffers for optional ZIP creation.
    """
    steps = 6
    images_base64 = []
    image_buffers = []

    with torch.no_grad():
        noise = noise.to(next(generator.parameters()).device)
        for i in range(num_images):
            alpha = 1.0  # Set alpha to 1.0 for full fade-in
            img_tensor = generator(noise[i:i+1], alpha, steps).squeeze(0)  # Generate an image
            img_tensor = (img_tensor * 0.5 + 0.5).clamp(0, 1)  # Normalize to [0, 1]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

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

