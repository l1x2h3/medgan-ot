# ===== Cell 0 =====
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from math import log2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import glob
import re

# 【新增】导入自定义评估工具
# 确保 metrics_tool.py 在同一目录下
try:
    from metrics_tool import ImageMetricsEvaluator
    HAS_METRICS_TOOL = True
except ImportError:
    HAS_METRICS_TOOL = False
    print("Warning: metrics_tool.py not found. Advanced metrics (LPIPS, VIF, UQI) will be skipped.")

# ===== Cell 1 =====
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===== Cell 2: Config & Args =====
DATASET_DEFAULT = r"data/archive/Training"
START_TRAIN_AT_IMG_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16] 
image_size_target = 128
CHANNELS_IMG = 3
Z_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)

def parse_args():
    parser = argparse.ArgumentParser(description="ProGAN Training and Testing")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--data_path', type=str, default=DATASET_DEFAULT, help='Path to training dataset')
    parser.add_argument('--test_data_path', type=str, default='data/archive/Testing', help='Path to testing dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join('artifacts','progan','checkpoints'), help='Directory to save/load models')
    parser.add_argument('--output_dir', type=str, default=os.path.join('artifacts','progan','samples'), help='Directory to save test results')
    parser.add_argument('--test_step', type=int, default=None, help='Specific step to test. If None, uses latest.')
    parser.add_argument('--num_test_images', type=int, default=32, help='Number of images to generate in test mode')
    
    import sys
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args([])
    return args

args = parse_args()
DATASET = args.data_path
CHECKPOINT_DIR = args.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if args.mode == 'test':
    os.makedirs(args.output_dir, exist_ok=True)

print(f"Running in [{args.mode.upper()}] mode on {DEVICE}")
print(f"Data Path: {DATASET}")
if args.mode == 'test':
    print(f"Test Data Path: {args.test_data_path}")
    print(f"Output Dir: {args.output_dir}")

seed_everything()

# ===== Cell 3 =====
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

# ===== Cell 4 =====
def get_loader(image_size, data_root=None):
    root = data_root if data_root else DATASET
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset path not found: {root}")
        
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ])
    try:
        batch_idx = int(log2(image_size / 4))
        if batch_idx >= len(BATCH_SIZES): batch_idx = len(BATCH_SIZES) - 1
        batch_size = BATCH_SIZES[batch_idx]
    except:
        batch_size = 16

    dataset = datasets.ImageFolder(root=root, transform=transform)
    if len(dataset) == 0: raise ValueError(f"No images found in {root}")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, dataset

# ===== Cell 5 (Check Loader - Optional visualization) =====
def check_loader():
    # Skipped in script mode to avoid blocking, can be enabled if needed
    pass

# ===== Cell 6-10 (Models: WSConv2d, PixelNorm, ConvBlock, Generator, Discriminator) =====
# Keeping original implementation unchanged for brevity, assuming they are present as in previous prompt
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

class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(PixelNorm(), nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0), nn.LeakyReLU(0.2),
                                     WSConv2d(in_channels, in_channels, 3, 1, 1), nn.LeakyReLU(0.2), PixelNorm())
        self.initial_rgb = WSConv2d(in_channels, img_channels, 1, 1, 0)
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])
        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, 1, 1, 0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        out = self.initial(x)
        if steps == 0: return self.initial_rgb(out)
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled)
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)

class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in, 1, 1, 0))
        self.initial_rgb = WSConv2d(img_channels, in_channels, 1, 1, 0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, 3, 1, 1), nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, 4, 1, 0), nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, 1, 1, 0)
        )
    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled
    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)
    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)

# ===== Cell 11 (Gradient Penalty) =====
def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)
    mixed_scores = critic(interpolated_images, alpha, train_step)
    gradient = torch.autograd.grad(inputs=interpolated_images, outputs=mixed_scores,
                                   grad_outputs=torch.ones_like(mixed_scores),
                                   create_graph=True, retain_graph=True)[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)

# ===== Cell 12 (Generate Examples) =====
def generate_examples(gen, steps, n=100, save_dir=None):
    gen.eval()
    alpha = 1.0
    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir)
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIM, 1, 1).to(DEVICE)
            img = gen(noise, alpha, steps)
            if save_dir: save_image(img*0.5+0.5, os.path.join(save_dir, f"img_{i}.png"))
    gen.train()

# ===== Utility: Save & Load =====
def save_progan_models(step, gen, critic, checkpoint_dir):
    path_gen = os.path.join(checkpoint_dir, f"gen_step{step}.pth")
    path_crit = os.path.join(checkpoint_dir, f"crit_step{step}.pth")
    torch.save({'step': step, 'gen_state_dict': gen.state_dict()}, path_gen)
    torch.save({'step': step, 'crit_state_dict': critic.state_dict()}, path_crit)
    print(f"Saved models for Step {step}")

def get_latest_step_checkpoint(checkpoint_dir, target_step=None):
    if target_step is not None:
        path = os.path.join(checkpoint_dir, f"gen_step{target_step}.pth")
        if os.path.exists(path): return path, target_step
        return None, None
    files = glob.glob(os.path.join(checkpoint_dir, "gen_step*.pth"))
    if not files: return None, None
    def extract_step(fname):
        match = re.search(r"step(\d+)\.pth", fname)
        return int(match.group(1)) if match else -1
    files.sort(key=extract_step)
    return files[-1], extract_step(files[-1])

def load_progan_checkpoint(path, gen, critic=None):
    checkpoint = torch.load(path, map_location=DEVICE)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    step = checkpoint.get('step', 0)
    if critic:
        crit_path = path.replace("gen_", "crit_")
        if os.path.exists(crit_path):
            critic.load_state_dict(torch.load(crit_path, map_location=DEVICE)['crit_state_dict'])
    print(f"Loaded Step {step} from {path}")
    return step

# ===== Cell 14 (Train Fn) =====
def train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(DEVICE)
        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, device=DEVICE)
        loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp + (0.001 * torch.mean(critic_real ** 2)))
        critic.zero_grad(); loss_critic.backward(); opt_critic.step()
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad(); loss_gen.backward(); opt_gen.step()
        alpha += cur_batch_size / ((PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)
        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())
    return alpha

# ===== Main Execution =====
if __name__ == "__main__":
    gen = Generator(Z_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    critic = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    if args.mode == 'train':
        print("--- Starting ProGAN Training ---")
        gen.train(); critic.train()
        step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
        max_steps = len(PROGRESSIVE_EPOCHS)
        for num_epochs in PROGRESSIVE_EPOCHS:
            if step >= max_steps: break
            alpha = 1e-5
            current_img_size = 4 * 2 ** step
            print(f"\nTraining Step {step}: Size {current_img_size}")
            try:
                loader, dataset = get_loader(current_img_size)
            except Exception as e:
                print(f"Error: {e}"); break
            for epoch in range(num_epochs):
                print(f"  Epoch [{epoch+1}/{num_epochs}]")
                alpha = train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen)
            save_progan_models(step, gen, critic, CHECKPOINT_DIR)
            generate_examples(gen, step, n=16, save_dir=os.path.join("saved_examples", f"step{step}"))
            step += 1

    elif args.mode == 'test':
        print("--- Starting ProGAN Testing with Full Metrics ---")
        if not HAS_METRICS_TOOL:
            print("ERROR: Cannot run full metrics test without metrics_tool.py. Please ensure the file exists and dependencies are installed.")
            exit(1)

        target_step = args.test_step
        if target_step is None:
            _, max_step = get_latest_step_checkpoint(CHECKPOINT_DIR)
            if max_step is None: raise FileNotFoundError("No checkpoints found.")
            target_step = max_step
            print(f"Using latest step: {target_step}")
        
        ckpt_path, loaded_step = get_latest_step_checkpoint(CHECKPOINT_DIR, target_step)
        if not ckpt_path: raise FileNotFoundError(f"Step {target_step} not found.")
        
        load_progan_checkpoint(ckpt_path, gen, critic)
        gen.eval(); critic.eval()
        
        img_size = 4 * 2 ** loaded_step
        test_loader = None
        real_source_name = ""
        
        # Load Test Data
        if os.path.exists(args.test_data_path):
            try:
                test_transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*CHANNELS_IMG, [0.5]*CHANNELS_IMG)
                ])
                test_dataset = datasets.ImageFolder(root=args.test_data_path, transform=test_transform)
                if len(test_dataset) > 0:
                    test_loader = DataLoader(test_dataset, batch_size=args.num_test_images, shuffle=False)
                    real_source_name = "Testing Folder"
            except Exception as e:
                print(f"Load test failed: {e}")
        
        if test_loader is None:
            print("Fallback to Training data for metrics.")
            test_loader, _ = get_loader(img_size, data_root=DATASET)
            real_source_name = "Training Folder (Fallback)"

        # Generate
        print(f"Generating {args.num_test_images} images...")
        with torch.no_grad():
            noise = torch.randn(args.num_test_images, Z_DIM, 1, 1).to(DEVICE)
            fake = gen(noise, 1.0, loaded_step)
            save_image(fake * 0.5 + 0.5, os.path.join(args.output_dir, f"grid_step{loaded_step}.png"), nrow=4)
            for i, img in enumerate(fake):
                save_image(img * 0.5 + 0.5, os.path.join(args.output_dir, f"gen_{i:04d}.png"))

        # Calculate Metrics
        if test_loader:
            real_batch, _ = next(iter(test_loader))
            count = min(real_batch.shape[0], fake.shape[0])
            real_sample = real_batch[:count].to(DEVICE)
            fake_sample = fake[:count]
            
            print(f"\nCalculating 6 Metrics (vs {real_source_name})...")
            evaluator = ImageMetricsEvaluator(device=DEVICE)
            
            results = evaluator.evaluate_batch(fake_sample, real_sample)
            
            print("\n" + "="*40)
            print(f"{'Metric':<10} | {'Value':>10}")
            print("="*40)
            for k, v in results.items():
                if k == "LPIPS" or k == "SSIM" or k == "UQI" or k == "VIF":
                    print(f"{k:<10} | {v:>10.4f}")
                elif k == "PSNR(dB)":
                    print(f"{k:<10} | {v:>10.2f}")
                else:
                    print(f"{k:<10} | {v:>10.2f}")
            print("="*40)
            
            # Save results to text file
            with open(os.path.join(args.output_dir, "metrics_results.txt"), 'w') as f:
                f.write(f"Step: {loaded_step}\nSource: {real_source_name}\n\n")
                for k, v in results.items():
                    f.write(f"{k}: {v}\n")
            print(f"Results saved to {os.path.join(args.output_dir, 'metrics_results.txt')}")
        else:
            print("Skipping metrics (no data).")
        
        print("\nTest completed.")
