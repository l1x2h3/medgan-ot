import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm


class Generator(nn.Module):
    def __init__(self, z_dim: int, features_g: int, channels_img: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_g * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 16, features_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, channels_img: int, features_d: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels_img, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 8, features_d * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def initialize_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def ot_projection(images: torch.Tensor, pool_size: int) -> torch.Tensor:
    pooled = torch.nn.functional.adaptive_avg_pool2d(images, (pool_size, pool_size))
    return pooled.flatten(start_dim=1)


def sinkhorn_ot_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 50,
) -> torch.Tensor:
    batch_size = x.size(0)
    cost = torch.cdist(x, y, p=2).pow(2)
    cost = cost / (cost.detach().mean() + 1e-8)

    mu = torch.full((batch_size,), 1.0 / batch_size, device=x.device, dtype=x.dtype)
    nu = torch.full((batch_size,), 1.0 / batch_size, device=x.device, dtype=x.dtype)

    kernel = torch.exp(-cost / epsilon).clamp_min(1e-8)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(n_iters):
        u = mu / (kernel @ v + 1e-8)
        v = nu / (kernel.t() @ u + 1e-8)

    transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    return torch.sum(transport * cost)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DCGAN with OT regularization.")
    parser.add_argument("--data-root", type=str, default="data/archive/Training")
    parser.add_argument("--out-dir", type=str, default="artifacts/ot/dcgan")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=200)
    parser.add_argument("--channels-img", type=int, default=3)
    parser.add_argument("--features-d", type=int, default=128)
    parser.add_argument("--features-g", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ot-lambda", type=float, default=0.5)
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-iters", type=int, default=50)
    parser.add_argument("--ot-pool-size", type=int, default=16)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    sample_dir = out_dir / "samples"
    ckpt_dir = out_dir / "checkpoints"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.ImageFolder(args.data_root, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    gen = Generator(args.z_dim, args.features_g, args.channels_img).to(device)
    disc = Discriminator(args.channels_img, args.features_d).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)

    global_step = 0
    for epoch in range(args.epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for real, _ in loop:
            real = real.to(device, non_blocking=True)
            bsz = real.size(0)

            noise = torch.randn(bsz, args.z_dim, 1, 1, device=device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            loss_disc_real = bce(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = bce(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = 0.5 * (loss_disc_real + loss_disc_fake)

            opt_disc.zero_grad(set_to_none=True)
            loss_disc.backward()
            opt_disc.step()

            noise_g = torch.randn(bsz, args.z_dim, 1, 1, device=device)
            fake_g = gen(noise_g)
            output = disc(fake_g).reshape(-1)
            loss_adv = bce(output, torch.ones_like(output))

            real_ot = ot_projection(real, args.ot_pool_size)
            fake_ot = ot_projection(fake_g, args.ot_pool_size)
            loss_ot = sinkhorn_ot_cost(
                fake_ot,
                real_ot,
                epsilon=args.ot_epsilon,
                n_iters=args.ot_iters,
            )

            loss_gen = loss_adv + args.ot_lambda * loss_ot

            opt_gen.zero_grad(set_to_none=True)
            loss_gen.backward()
            opt_gen.step()

            if global_step % args.sample_every == 0:
                with torch.no_grad():
                    fake_fixed = gen(fixed_noise)
                    grid = vutils.make_grid(fake_fixed[:32], normalize=True, nrow=8)
                    vutils.save_image(grid, sample_dir / f"step_{global_step:07d}.png")

            loop.set_postfix(
                loss_d=f"{loss_disc.item():.4f}",
                loss_g=f"{loss_gen.item():.4f}",
                adv=f"{loss_adv.item():.4f}",
                ot=f"{loss_ot.item():.4f}",
            )
            global_step += 1

        torch.save(
            {
                "epoch": epoch,
                "gen": gen.state_dict(),
                "disc": disc.state_dict(),
                "opt_gen": opt_gen.state_dict(),
                "opt_disc": opt_disc.state_dict(),
                "args": vars(args),
            },
            ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
        )


if __name__ == "__main__":
    main()
