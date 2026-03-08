# Auto-generated from notebook.
# Source: D:/homework/研一下/paper_ImageTranslation/medgan-ot/notebooks/progan/notebook-pro.ipynb
# Note: Review dataset/save paths before running.

# ===== Cell 0 =====
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from math import log2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===== Cell 1 =====
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

# ===== Cell 2 =====
DATASET = r"C:\Users\mzlwm\OneDrive\Desktop\MEDGAN\dataset"
START_TRAIN_AT_IMG_SIZE = 4
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE           = 1e-3
BATCH_SIZES             = [32, 32, 32, 16, 16, 16] #you can use [32, 32, 32, 16, 16, 16, 16, 8, 4] for example if you want to train until 1024x1024, but again this numbers depend on your vram
image_size              = 128
CHANNELS_IMG            = 3
Z_DIM                   = 256  # should be 512 in original paper
IN_CHANNELS             = 256  # should be 512 in original paper
LAMBDA_GP               = 10
PROGRESSIVE_EPOCHS      = [30] * len(BATCH_SIZES)
#PROGRESSIVE_EPOCHS = [5] * len(BATCH_SIZES)  # Set 5 epochs for testing

# ===== Cell 3 =====
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

# ===== Cell 4 =====
def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ]
    )
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return loader, dataset

# ===== Cell 5 =====
def check_loader():
    loader,_ = get_loader(128)
    cloth ,_ = next(iter(loader))
    _, ax    = plt.subplots(3,3, figsize=(8,8))
    plt.suptitle('Some real samples', fontsize=15, fontweight='bold')
    ind = 0
    for k in range(3):
        for kk in range(3):
            ind += 1
            ax[k][kk].imshow((cloth[ind].permute(1,2,0)+1)/2)
check_loader()

# ===== Cell 6 =====
class WSConv2d(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
    ):
        super(WSConv2d, self).__init__()
        self.conv      = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale     = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias      = self.conv.bias #Copy the bias of the current column layer
        self.conv.bias = None      #Remove the bias

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

# ===== Cell 7 =====
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

# ===== Cell 8 =====
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm
        self.conv1  = WSConv2d(in_channels, out_channels)
        self.conv2  = WSConv2d(out_channels, out_channels)
        self.leaky  = nn.LeakyReLU(0.2)
        self.pn     = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn else x
        return x

# ===== Cell 9 =====
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()

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

        for i in range(
            len(factors) - 1
        ):  # -1 to prevent index error because of factors[i+1]
            conv_in_c  = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)

        # The number of channels in upscale will stay the same, while
        # out which has moved through prog_blocks might change. To ensure
        # we can convert both to rgb we use different rgb_layers
        # (steps-1) and steps for upscaled, out respectively
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

# ===== Cell 10 =====
class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # here we work back ways from factors because the discriminator
        # should be mirrored from the generator. So the first prog_block and
        # rgb layer we append will work for input size 1024x1024, then 512->256-> etc
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # perhaps confusing name "initial_rgb" this is just the RGB layer for 4x4 input size
        # did this to "mirror" the generator initial_rgb
        self.initial_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        # this is the block for 4x4 input size
        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            ),  # we use this instead of linear layer
        )

    def fade_in(self, alpha, downscaled, out):
        """Used to fade in downscaled using avg pooling and output from CNN"""
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        # where we should start in the list of prog_blocks, maybe a bit confusing but
        # the last is for the 4x4. So example let's say steps=1, then we should start
        # at the second to last because input_size will be 8x8. If steps==0 we just
        # use the final block
        cur_step = len(self.prog_blocks) - steps

        # convert from rgb as initial step, this will depend on
        # the image size (each will have it's on rgb layer)
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:  # i.e, image is 4x4
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # because prog_blocks might change the channels, for down scale we use rgb_layer
        # from previous/smaller size which in our case correlates to +1 in the indexing
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        # the fade_in is done first between the downscaled and the input
        # this is opposite from the generator
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

# ===== Cell 11 =====
def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

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

# ===== Cell 12 =====
def generate_examples(gen, steps, n=100):

    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)
            img = gen(noise, alpha, steps)
            if not os.path.exists(f'saved_examples/step{steps}'):
                os.makedirs(f'saved_examples/step{steps}')
            save_image(img*0.5+0.5, f"saved_examples/step{steps}/img_{i}.png")
    gen.train()

# ===== Cell 13 =====
torch.backends.cudnn.benchmarks = True

# ===== Cell 14 =====
def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)

        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, device=DEVICE)
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_GP * gp
            + (0.001 * torch.mean(critic_real ** 2))
        )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )


    return alpha

# ===== Cell 15 =====
# initialize gen and disc, note: discriminator we called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(
    Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG
).to(DEVICE)
critic = Discriminator(
    IN_CHANNELS, img_channels=CHANNELS_IMG
).to(DEVICE)

# initialize optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic = optim.Adam(
    critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99)
)


gen.train()
critic.train()

step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
for num_epochs in PROGRESSIVE_EPOCHS:
    alpha = 1e-5  # start with very low alpha, you can start with alpha=0
    loader, dataset = get_loader(4 * 2 ** step)  # 4->0, 8->1, 16->2, 32->3, 64 -> 4
    print(f"Current image size: {4 * 2 ** step}")

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        alpha = train_fn(
            critic,
            gen,
            loader,
            dataset,
            step,
            alpha,
            opt_critic,
            opt_gen,
        )
    generate_examples(gen, step, n=100)


    step += 1  # progress to the next img size

# ===== Cell 16 =====
# prompt: save the madel as pht file and h5 and pkl 

import torch
import os

# Assuming 'gen' and 'critic' are your trained generator and critic models
# ... (Your existing code)

# Save the model as a .pth file
torch.save(gen.state_dict(), 'generator.pth')
torch.save(critic.state_dict(), 'critic.pth')


# Save the model as an .h5 file (requires h5py library)
import h5py

with h5py.File('generator.h5', 'w') as hf:
    for k, v in gen.state_dict().items():
        hf.create_dataset(k, data=v.cpu().numpy())

with h5py.File('critic.h5', 'w') as hf:
    for k, v in critic.state_dict().items():
        hf.create_dataset(k, data=v.cpu().numpy())

# ===== Cell 17 =====
# prompt: visualize generated images from step 5

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Specify the directory containing the generated images
image_dir = "saved_examples/step5"  # Replace with the actual directory


# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Create a figure and axes for the subplots
fig, axes = plt.subplots(10, 10, figsize=(20, 20))  # Adjust the number of rows and columns as needed

# Iterate over the image files and display them in the subplots
for i, image_file in enumerate(image_files[:100]):  # Display up to 100 images
    if i >= 100:
        break
    img = mpimg.imread(os.path.join(image_dir, image_file))
    row = i // 10
    col = i % 10
    axes[row, col].imshow(img)
    axes[row, col].axis('off')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# ===== Cell 18 =====
import pickle

# Save generator
with open('generator.pkl', 'wb') as f:
    pickle.dump(gen.state_dict(), f)

# Save critic
with open('critic.pkl', 'wb') as f:
    pickle.dump(critic.state_dict(), f)

# ===== Cell 19 =====
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Directory containing generated images
generated_image_dir = "saved_examples/step5"  # Replace with your actual directory

# Load real examples
real_loader, _ = get_loader(128)  # Assuming 128x128 real images
real_images, _ = next(iter(real_loader))

# Load generated images
generated_image_files = [f for f in os.listdir(generated_image_dir) if os.path.isfile(os.path.join(generated_image_dir, f))]

# Create a figure for side-by-side comparison
fig, axes = plt.subplots(2, 6, figsize=(18, 6))  # 2 rows, 6 columns (left: real, right: generated)

# Display real images on the right
for i in range(6):
    axes[0, i].imshow((real_images[i].permute(1, 2, 0) + 1) / 2)  # Rescale from [-1, 1] to [0, 1]
    axes[0, i].set_title("Real Image")
    axes[0, i].axis("off")

# Display generated images on the left
for i in range(6):
    if i >= len(generated_image_files):  # Handle cases with fewer generated images
        break
    img = mpimg.imread(os.path.join(generated_image_dir, generated_image_files[i]))
    axes[1, i].imshow(img)
    axes[1, i].set_title("Generated Image")
    axes[1, i].axis("off")

# Adjust layout and display the plot
plt.tight_layout()
#plt.suptitle("Real vs Generated Images", fontsize=16, fontweight="bold")
plt.show()

# ===== Cell 20 =====
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Function to calculate Dice Score
def calculate_dice_score(img1, img2):
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    intersection = np.sum(img1_flat * img2_flat)
    return (2.0 * intersection) / (np.sum(img1_flat) + np.sum(img2_flat))

# Function to calculate SSIM
def calculate_ssim(img1, img2):
    return ssim(img1, img2)

# Directories containing real and generated images
real_images_dir = "real_examples"  # Adjust path as necessary
generated_images_dir = "saved_examples/step5"  # Adjust path as necessary

# Initialize total metrics
total_psnr = 0.0
total_ssim = 0.0
total_dice = 0.0
num_images = 0

# Process images from both directories
for i in range(100):  # Assuming there are 100 images (0000.jpg to 0099.jpg)
    # Define the paths for the real and generated images
    real_image_path = os.path.join(real_images_dir, f"{i:04d}.jpg")  # Real image path
    generated_image_path = os.path.join(generated_images_dir, f"img_{i}.png")  # Generated image path

    # Check if both files exist
    if os.path.exists(real_image_path) and os.path.exists(generated_image_path):
        # Load the real image (in grayscale) and resize to 128x128
        real_image = cv2.imread(real_image_path, cv2.IMREAD_GRAYSCALE)
        real_image = cv2.resize(real_image, (128, 128))

        # Load the generated image (in grayscale)
        generated_image = cv2.imread(generated_image_path, cv2.IMREAD_GRAYSCALE)
        generated_image = cv2.resize(generated_image, (128, 128))

        # Calculate metrics
        psnr_value = calculate_psnr(real_image, generated_image)
        ssim_value = calculate_ssim(real_image, generated_image)
        dice_value = calculate_dice_score(real_image, generated_image)

        # Accumulate metrics
        total_psnr += psnr_value
        total_ssim += ssim_value
        total_dice += dice_value
        num_images += 1

# Calculate averages
if num_images > 0:
    average_psnr = total_psnr / num_images
    average_ssim = total_ssim / num_images
    average_dice = total_dice / num_images

    # Display results
    print(f"Processed {num_images} images.")
    print(f"Average PSNR: {average_psnr:.2f}")
    print(f"Average SSIM: {average_ssim:.2f}")
    print(f"Average Dice Score: {average_dice:.2f}")
else:
    print("No valid images found to process.")

