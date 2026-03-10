import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple
import json

import os
import sys
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_py.metrics_tool import ImageMetricsEvaluator
from ot_experiments.dcgan_ot.train_dcgan_ot import Generator


def list_checkpoints(checkpoint_dir: Path) -> Iterable[Path]:
    checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
    checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return checkpoints


def load_checkpoint(path: Path, device: torch.device) -> Optional[dict]:
    try:
        return torch.load(path, map_location=device)
    except (RuntimeError, EOFError) as exc:
        print(f"Checkpoint {path} is unreadable ({exc}); trying an earlier checkpoint.")
        return None


def load_latest_valid_checkpoint(
    checkpoint_dir: Path, device: torch.device
) -> Tuple[Optional[Path], Optional[dict]]:
    for checkpoint in reversed(list(list_checkpoints(checkpoint_dir))):
        data = load_checkpoint(checkpoint, device)
        if data is not None:
            return checkpoint, data
    return None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test DCGAN+OT checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/ot/dcgan/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ot/dcgan/tests"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--z-dim", type=int, default=200)
    parser.add_argument("--features-g", type=int, default=128)
    parser.add_argument("--channels-img", type=int, default=3)
    parser.add_argument("--test-data-root", type=Path, default=Path("data/archive/Testing"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--metrics-output", type=Path, default=Path("artifacts/ot/dcgan/tests/metrics.json"))
    parser.add_argument("--skip-metrics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path, checkpoint = load_latest_valid_checkpoint(args.checkpoint_dir, device)
    if checkpoint_path is None:
        print(
            f"Unable to load a valid checkpoint from {args.checkpoint_dir}; continuing with random weights."
        )
    used_seed = checkpoint_path.stem if checkpoint_path else "random"

    def build_test_loader(root: Path) -> DataLoader:
        transform = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.ImageFolder(root, transform=transform)
        return DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    def collect_fake_real_batches(
        loader: DataLoader, max_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fake_batches = []
        real_batches = []
        collected = 0
        for real_batch, _ in loader:
            if collected >= max_samples:
                break
            real_batch = real_batch.to(device)
            remaining = max_samples - collected
            real_batch = real_batch[:remaining]
            if real_batch.size(0) == 0:
                break
            noise = torch.randn(real_batch.size(0), args.z_dim, 1, 1, device=device)
            fake_batch = gen(noise)
            fake_batches.append(fake_batch.detach().cpu())
            real_batches.append(real_batch.detach().cpu())
            collected += real_batch.size(0)
        if collected == 0:
            return torch.empty(0, device="cpu"), torch.empty(0, device="cpu")
        return torch.cat(fake_batches, dim=0), torch.cat(real_batches, dim=0)

    gen = Generator(args.z_dim, args.features_g, args.channels_img).to(device)
    if checkpoint is not None:
        gen.load_state_dict(checkpoint["gen"])
    gen.eval()

    with torch.no_grad():
        noise = torch.randn(args.num_samples, args.z_dim, 1, 1, device=device)
        samples = gen(noise)

    grid = vutils.make_grid(samples, normalize=True, nrow=4)
    output_path = args.output_dir / f"dcgan_ot_test_{used_seed}.png"
    vutils.save_image(grid, output_path)
    print(f"Generated grid saved to {output_path}")

    if not args.skip_metrics:
        fake_tensors = torch.empty(0, device="cpu")
        real_tensors = torch.empty(0, device="cpu")
        if not args.test_data_root.exists():
            print(f"Test data root {args.test_data_root} not found; skipping metrics evaluation.")
        else:
            metrics_loader = build_test_loader(args.test_data_root)
            fake_tensors, real_tensors = collect_fake_real_batches(metrics_loader, args.max_eval_samples)
        if fake_tensors.size(0) > 0 and real_tensors.size(0) > 0:
            evaluator = ImageMetricsEvaluator(device=device)
            metrics = evaluator.evaluate_batch(fake_tensors, real_tensors)
            metrics["checkpoint"] = used_seed
            metrics["samples"] = fake_tensors.size(0)
            print("Evaluation metrics:", metrics)
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.metrics_output, "w", encoding="utf-8") as outf:
                json.dump(metrics, outf, indent=2, ensure_ascii=False)
            print(f"Metrics written to {args.metrics_output}")
        else:
            print("No samples collected for metrics evaluation; skipping metrics dump.")


if __name__ == "__main__":
    main()
