import io
import torch
import base64
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Generator_WGAN(nn.Module):
    def __init__(self, z_dim=256, img_channels=1, features_g=32):
        super(Generator_WGAN, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 32, 4, 2, 0),
            self._block(features_g * 32, features_g * 16, 4, 2, 1),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            self._block(features_g * 2, features_g, 4, 2, 1),
            nn.ConvTranspose2d(features_g, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)

# Function to generate WGAN images
def generate_examples_WGAN(generator, noise, num_images):
    images_base64 = []
    image_buffers = []

    with torch.no_grad():
        generated_images = generator(noise)
        generated_images = (generated_images + 1) / 2  # Normalize to [0, 1]

        for i in range(num_images):
            img_tensor = generated_images[i].cpu().squeeze(0)
            img_np = (img_tensor.numpy() * 255).astype('uint8')

            # Convert to Base64-encoded image
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(img_np, cmap='gray')
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

