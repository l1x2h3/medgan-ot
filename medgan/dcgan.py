import io
import torch
import base64
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Define the Generator class for DCGAN
class Generator_DCGAN(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator_DCGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


def generate_examples_DCGAN_1(generator, z_dim, class_name="Example"):
    """
    Generate an example using a DCGAN generator and visualize it.

    Args:
        generator (torch.nn.Module): The trained DCGAN generator model.
        z_dim (int): Dimension of the latent noise vector.
        class_name (str): Class label for the generated images.

    Returns:
        matplotlib.figure.Figure: A matplotlib figure showing the generated image.
    """
    generator.eval()
    noise = torch.randn(1, z_dim, 1, 1).to(torch.device('cpu'))

    with torch.no_grad():
        generated_image = generator(noise)

    # Display the generated image
    fig, ax = plt.subplots(figsize=(2, 2))
    fig.patch.set_facecolor('white')  # Set the figure's background color
    img_to_display = (generated_image[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    ax.imshow(img_to_display)  # Convert tensor to HxWxC for plt.imshow
    ax.axis('off')  # Turn off axis
    ax.set_facecolor('white')  # Set the axis background color
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove space around image

    return fig



def generate_examples_DCGAN(generator, noise, num_images):
    """
    Generate images using a DCGAN generator and return them as Base64-encoded strings.

    Args:
        generator (torch.nn.Module): Pre-trained DCGAN generator model.
        noise (torch.Tensor): Random noise tensor for image generation.
        num_images (int): Number of images to generate.

    Returns:
        list: Base64-encoded images.
        list: BytesIO image buffers for optional ZIP creation.
    """
    images_base64 = []
    image_buffers = []

    with torch.no_grad():
        generated_images = generator(noise)
        for i in range(num_images):
            img_tensor = (generated_images[i] + 1) / 2  # Normalize to [0, 1]
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


