# Auto-generated from notebook.
# Source: D:/homework/研一下/paper_ImageTranslation/medgan-ot/notebooks/dcgan/pt/notebook-dc.ipynb
# Note: Review dataset/save paths before running.

# ===== Cell 1 =====
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import numpy as np

# ===== Cell 2 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 64
IMAGE_SIZE = 128
CHANNELS_IMG = 3
Z_DIM = 200
NUM_EPOCHS = 100
FEATURES_DISC = 128
FEATURES_GEN = 128

# ===== Cell 3 =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset = datasets.ImageFolder('data/archive/Training', transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== Cell 4 =====
real_batch = next(iter(dataloader))
plt.figure(figsize=(7, 7))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:49], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.close()

# ===== Cell 5 =====
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input: Z latent vector into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # State size: (ngf * 16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf * 8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf * 4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf * 2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size: (nc) x 128 x 128
        )

    def forward(self, x):
        return self.main(x)

# ===== Cell 6 =====
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf * 2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf * 4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf * 8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf * 16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output size: 1
        )

    def forward(self, x):
        return self.main(x)

# ===== Cell 7 =====
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# ===== Cell 8 =====
gen = Generator(1, Z_DIM, FEATURES_GEN, CHANNELS_IMG).to(device)

disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

# ===== Cell 9 =====
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

# ===== Cell 10 =====
SHOW_IMAGES = False

def show_tensor_images(image_tensor, num_images=32, size=(1, 64, 64)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.figure()
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if SHOW_IMAGES:
        # Non-blocking display, then immediately close.
        plt.show(block=False)
        plt.pause(0.001)
    plt.close()

# ===== Cell 11 =====
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _ ) in enumerate(dataloader):
        real = real.to(device)
        ### create noise tensor
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        ### Print losses occasionally and fake images occasionally
        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                show_tensor_images(img_grid_fake)

# ===== Cell 12 =====
# prompt: plot a new generated images

import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Assuming 'fake' is the tensor of generated images from the last batch
# of the training loop, and 'fixed_noise' is a tensor of fixed noise.

with torch.no_grad():
    fake = gen(fixed_noise)  # Generate images using the fixed noise
    img_grid_fake = vutils.make_grid(fake[:32], normalize=True)
    show_tensor_images(img_grid_fake)

# ===== Cell 13 =====
# prompt: calculate the PSNR

import math

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Example usage (assuming 'real' and 'fake' are tensors of images):
with torch.no_grad():
    fake = gen(fixed_noise)
    # Assuming 'real' is a batch of real images from your dataloader
    # You need to make sure 'real' and 'fake' have the same shape and are on the same device
    # Convert 'real' to a PyTorch tensor and move it to the device
    real_sample = torch.tensor(real[0:fake.shape[0]], device=device, dtype=fake.dtype)

    psnr_value = psnr(real_sample, fake)
    print(f"PSNR: {psnr_value}")

# ===== Cell 14 =====
# prompt: CALCULATE THE SSIM

from skimage.metrics import structural_similarity as ssim
import numpy as np

# Assuming 'fake' is the tensor of generated images and 'real' is the tensor of real images
with torch.no_grad():
    fake = gen(fixed_noise)
    real_sample = torch.tensor(real[0:fake.shape[0]], device=device, dtype=fake.dtype)

    # Move tensors to CPU and convert to NumPy arrays
    fake_np = fake.cpu().numpy()
    real_np = real_sample.cpu().numpy()

    # Calculate SSIM for each image in the batch
    ssim_values = []
    for i in range(fake.shape[0]):
        # Reshape to (height, width, channels)
        fake_img = np.transpose(fake_np[i], (1, 2, 0))
        real_img = np.transpose(real_np[i], (1, 2, 0))

        # Ensure images are in the range [0, 1]
        fake_img = (fake_img + 1) / 2
        real_img = (real_img + 1) / 2

        # Calculate SSIM, explicitly setting win_size
        ssim_value = ssim(real_img, fake_img, multichannel=True, data_range=1, win_size=3)  # win_size=3 for smaller images
        ssim_values.append(ssim_value)

    # Calculate average SSIM across the batch
    avg_ssim = np.mean(ssim_values)
    print(f"Average SSIM: {avg_ssim}")

# ===== Cell 15 =====
# prompt: calculate the dice score

# Assuming 'fake' is the tensor of generated images and 'real' is the tensor of real images
with torch.no_grad():
    fake = gen(fixed_noise)
    real_sample = torch.tensor(real[0:fake.shape[0]], device=device, dtype=fake.dtype)

    # Move tensors to CPU and convert to NumPy arrays
    fake_np = fake.cpu().numpy()
    real_np = real_sample.cpu().numpy()

    # Calculate Dice score for each image in the batch
    dice_scores = []
    for i in range(fake.shape[0]):
        # Reshape to (height, width, channels)
        fake_img = np.transpose(fake_np[i], (1, 2, 0))
        real_img = np.transpose(real_np[i], (1, 2, 0))

        # Ensure images are in the range [0, 1]
        fake_img = (fake_img + 1) / 2
        real_img = (real_img + 1) / 2

        # Convert images to binary masks (you might need to adjust the threshold)
        fake_mask = (fake_img > 0.5).astype(np.uint8)
        real_mask = (real_img > 0.5).astype(np.uint8)

        # Calculate Dice score
        intersection = np.logical_and(fake_mask, real_mask).sum()
        dice = (2. * intersection) / (fake_mask.sum() + real_mask.sum())
        dice_scores.append(dice)

    # Calculate average Dice score across the batch
    avg_dice = np.mean(dice_scores)
    print(f"Average Dice Score: {avg_dice}")

# ===== Cell 16 =====
# prompt: calculate the FID

import numpy as np
from scipy.linalg import sqrtm

# Assuming 'fake' is the tensor of generated images and 'real' is the tensor of real images
with torch.no_grad():
    fake = gen(fixed_noise)
    real_sample = torch.tensor(real[0:fake.shape[0]], device=device, dtype=fake.dtype)

    # Move tensors to CPU and convert to NumPy arrays
    fake_np = fake.cpu().numpy()
    real_np = real_sample.cpu().numpy()

    # Calculate FID
    # 1. Calculate activations for real and fake images using a pre-trained Inception model
    # ... (Code to load and use Inception model) ...
    # Example placeholder for activations (replace with actual activations)
    act1 = np.random.rand(100, 2048)  # Replace with activations of real images
    act2 = np.random.rand(100, 2048)  # Replace with activations of fake images

    # 2. Calculate the mean and covariance of activations
    mu1 = np.mean(act1, axis=0)
    sigma1 = np.cov(act1, rowvar=False)
    mu2 = np.mean(act2, axis=0)
    sigma2 = np.cov(act2, rowvar=False)

    # 3. Calculate FID
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    print(f"FID: {fid}")
