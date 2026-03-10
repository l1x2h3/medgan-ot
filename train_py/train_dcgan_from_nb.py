# ===== Cell 1: Imports & Setup =====
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re

# ===== Cell 2: Hyperparameters & Config =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 128
CHANNELS_IMG = 3
Z_DIM = 200
NUM_EPOCHS = 100
FEATURES_DISC = 128
FEATURES_GEN = 128

SAVE_INTERVAL = 5
MAX_KEEP_MODELS = 2
CHECKPOINT_DIR = os.path.join("artifacts", "dcgan", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== Cell 3: Argument Parsing (已更新) =====
def parse_args():
    parser = argparse.ArgumentParser(description="DCGAN Training and Testing")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode to run: "train" or "test"')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory to load/save checkpoints')
    
    # 【修改点 1】默认训练数据路径
    parser.add_argument('--data_path', type=str, default='data/archive/Training',
                        help='Path to training data')
    
    # 【新增】测试数据路径参数
    parser.add_argument('--test_data_path', type=str, default='data/archive/Testing',
                        help='Path to testing data (used only in test mode)')
    
    parser.add_argument('--num_test_images', type=int, default=32,
                        help='Number of images to generate during testing')
    parser.add_argument('--output_dir', type=str, default=os.path.join('artifacts','dcgan','samples'),
                        help='Directory to save test results')
    
    # 兼容 Notebook 和 命令行
    # 如果在命令行运行，使用 parser.parse_args()
    # 如果在 Notebook 运行且未传参，使用 parse_args([]) 保持默认
    import sys
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args([])
    
    return args

args = parse_args()

# 更新全局配置
CHECKPOINT_DIR = args.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
if args.mode == 'test':
    os.makedirs(args.output_dir, exist_ok=True)

print(f"Running in [{args.mode.upper()}] mode.")
print(f"Device: {device}")
if args.mode == 'train':
    print(f"Training Data: {args.data_path}")
else:
    print(f"Training Data (for ref): {args.data_path}")
    print(f"Testing Data: {args.test_data_path}")

# ===== Cell 4: Data Loading (已更新) =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 1. 加载训练集 (训练模式必需，测试模式仅用于参考或不需要)
train_dataset = datasets.ImageFolder(args.data_path, transform=transform)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 【新增】2. 加载测试集 (仅在测试模式或需要时使用)
test_dataloader = None
if os.path.exists(args.test_data_path):
    test_dataset = datasets.ImageFolder(args.test_data_path, transform=transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False) # 测试时通常不shuffle
    print(f"Loaded {len(test_dataset)} images from Testing folder.")
else:
    print(f"Warning: Testing folder '{args.test_data_path}' not found. Will fallback to Training data for metrics if needed.")

# 预览训练数据
try:
    real_batch = next(iter(train_dataloader))
    plt.figure(figsize=(7, 7))
    plt.axis("off")
    plt.title("Training Images Sample")
    plt.imshow(np.transpose(make_grid(real_batch[0][:49], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.close()
except Exception as e:
    print(f"Could not load training batch: {e}")

# ===== Cell 5 & 6: Models Definition (保持不变) =====
class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
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
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)

# ===== Cell 7: Weight Initialization =====
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# ===== Cell 8: Initialize Models =====
gen = Generator(1, Z_DIM, FEATURES_GEN, CHANNELS_IMG).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

# ===== Cell 9: Optimizers & Fixed Noise =====
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(args.num_test_images, Z_DIM, 1, 1).to(device)

# ===== Utility: Load Latest Checkpoint =====
def get_latest_checkpoint(checkpoint_dir, prefix='gen_epoch_'):
    files = glob.glob(os.path.join(checkpoint_dir, f"{prefix}*.pth"))
    if not files:
        return None, -1
    def extract_epoch(fname):
        match = re.search(r"epoch_(\d+)\.pth", fname)
        return int(match.group(1)) if match else -1
    files.sort(key=extract_epoch)
    latest_file = files[-1]
    latest_epoch = extract_epoch(latest_file)
    return latest_file, latest_epoch

def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint.get('epoch', 0)

# ===== Utility: Save & Cleanup =====
saved_model_files = []

def save_and_cleanup(epoch, gen, disc, opt_gen, opt_disc, saved_files_list, max_keep=2):
    filename_gen = f"gen_epoch_{epoch}.pth"
    filename_disc = f"disc_epoch_{epoch}.pth"
    path_gen = os.path.join(CHECKPOINT_DIR, filename_gen)
    path_disc = os.path.join(CHECKPOINT_DIR, filename_disc)
    
    torch.save({'epoch': epoch, 'model_state_dict': gen.state_dict(), 'optimizer_state_dict': opt_gen.state_dict()}, path_gen)
    torch.save({'epoch': epoch, 'model_state_dict': disc.state_dict(), 'optimizer_state_dict': opt_disc.state_dict()}, path_disc)
    print(f"Saved models for epoch {epoch}")
    
    saved_files_list.append(path_gen)
    saved_files_list.append(path_disc)
    
    while len(saved_files_list) > max_keep * 2:
        file_to_remove = saved_files_list.pop(0)
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)

# ===== Cell 10: Visualization Helper =====
def show_tensor_images(image_tensor, num_images=32, save_path=None):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Image saved to {save_path}")
    plt.close()

# ===== MODE: TRAINING =====
if args.mode == 'train':
    print("--- Starting Training ---")
    gen.train()
    disc.train()
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real, _ ) in enumerate(train_dataloader): # 使用 train_dataloader
            real = real.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(train_dataloader)} Loss D: {loss_disc:.4f}, G: {loss_gen:.4f}")
        
        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_and_cleanup(epoch, gen, disc, opt_gen, opt_disc, saved_model_files, MAX_KEEP_MODELS)
    print("Training Finished.")

# ===== MODE: TESTING (已更新以使用 Testing 数据集) =====
elif args.mode == 'test':
    print("--- Starting Testing Mode ---")
    
    latest_gen_path, latest_epoch = get_latest_checkpoint(CHECKPOINT_DIR, prefix='gen_epoch_')
    latest_disc_path, _ = get_latest_checkpoint(CHECKPOINT_DIR, prefix='disc_epoch_')
    
    if latest_gen_path is None:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}.")
    
    print(f"Loading Model from Epoch {latest_epoch}...")
    load_checkpoint(latest_gen_path, gen)
    load_checkpoint(latest_disc_path, disc)
    
    gen.eval()
    disc.eval()
    
    # 1. 生成图像
    print(f"Generating {args.num_test_images} images...")
    with torch.no_grad():
        noise = torch.randn(args.num_test_images, Z_DIM, 1, 1).to(device)
        fake = gen(noise)
        
        grid_path = os.path.join(args.output_dir, f"generated_grid_epoch_{latest_epoch}.png")
        show_tensor_images(fake, num_images=args.num_test_images, save_path=grid_path)
        
        for i, img in enumerate(fake):
            img_path = os.path.join(args.output_dir, f"gen_{i:04d}.png")
            save_image((img + 1) / 2, img_path)

    # 2. 计算指标 (关键修改：优先使用 Testing 数据集)
    real_source_name = ""
    real_sample = None
    
    if test_dataloader is not None:
        # 【核心修改】如果有测试集，从测试集取数据
        try:
            real_batch, _ = next(iter(test_dataloader))
            real_sample = real_batch[:fake.shape[0]].to(device)
            real_source_name = "Testing Folder"
        except StopIteration:
            print("Warning: Testing dataloader is empty.")
    
    if real_sample is None:
        # 如果测试集不可用，回退到训练集（防止报错）
        print("Fallback: Using Training data for metric comparison as Testing data was unavailable.")
        real_batch, _ = next(iter(train_dataloader))
        real_sample = real_batch[:fake.shape[0]].to(device)
        real_source_name = "Training Folder (Fallback)"

    print(f"\nCalculating Metrics comparing Generated Images vs. Real Images from: {real_source_name}")
    
    try:
        # PSNR
        import math
        def calc_psnr(img1, img2):
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0: return 100
            return 20 * math.log10(1.0 / torch.sqrt(mse))
        
        fake_norm = (fake + 1) / 2
        real_norm = (real_sample + 1) / 2
        psnr_val = calc_psnr(real_norm, fake_norm)
        print(f"PSNR: {psnr_val:.4f}")
        
        # SSIM
        from skimage.metrics import structural_similarity as ssim
        fake_np = fake_norm.cpu().numpy().transpose(0, 2, 3, 1)
        real_np = real_norm.cpu().numpy().transpose(0, 2, 3, 1)
        
        ssim_scores = []
        for i in range(fake.shape[0]):
            score = ssim(real_np[i], fake_np[i], multichannel=True, data_range=1, win_size=3)
            ssim_scores.append(score)
        print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
        
        # Dice
        dice_scores = []
        for i in range(fake.shape[0]):
            f_mask = (fake_np[i] > 0.5).astype(np.uint8)
            r_mask = (real_np[i] > 0.5).astype(np.uint8)
            inter = np.logical_and(f_mask, r_mask).sum()
            union = f_mask.sum() + r_mask.sum()
            dice = (2. * inter) / union if union > 0 else 0
            dice_scores.append(dice)
        print(f"Average Dice: {np.mean(dice_scores):.4f}")
        
        print("\nFID: Requires external library (pytorch-fid).")
        print(f"Command: python -m pytorch_fid {args.test_data_path} {args.output_dir}")

    except Exception as e:
        print(f"Error calculating metrics: {e}")

    print(f"\nTest completed. Results saved to: {args.output_dir}")
