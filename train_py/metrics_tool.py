# metrics_tool.py
import torch
import numpy as np
import cv2
try:
    from scipy import misc as scipy_misc
except ImportError:
    scipy_misc = None
else:
    if not hasattr(scipy_misc, "imresize"):
        from PIL import Image

        def _imresize(image, size, interp="bilinear"):
            if isinstance(size, (list, tuple)):
                height, width = size
            else:
                height = width = size
            resample_map = {
                "nearest": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC,
                "lanczos": Image.LANCZOS,
            }
            resample = resample_map.get(interp, Image.BILINEAR)
            img = Image.fromarray(image.astype(np.uint8))
            resized = img.resize((width, height), resample=resample)
            return np.array(resized)

        scipy_misc.imresize = _imresize
from sewar.full_ref import mse as sewar_mse, psnr as sewar_psnr, ssim as sewar_ssim, uqi as sewar_uqi
from piq import vif_p
import lpips
from tqdm import tqdm


class ImageMetricsEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        print("Initializing LPIPS model...")
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        self.loss_fn_alex.eval()
        print("LPIPS model ready.")

    def preprocess(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Converts tensor in [-1, 1] to numpy uint8 in [0, 255] with shape (N, H, W, C)
        """
        img_tensor = torch.clamp(img_tensor, -1, 1)
        img_np = ((img_tensor + 1) / 2).detach().cpu().numpy()
        img_np = np.transpose(img_np, (0, 2, 3, 1))
        return (img_np * 255).astype(np.uint8)

    def calculate_vif(self, fake_tensor: torch.Tensor, real_tensor: torch.Tensor) -> float:
        fake_norm = (fake_tensor.clamp(-1, 1) + 1) / 2
        real_norm = (real_tensor.clamp(-1, 1) + 1) / 2
        with torch.no_grad():
            return float(
                vif_p(
                    fake_norm.to(self.device),
                    real_norm.to(self.device),
                    data_range=1.0,
                )
                .cpu()
                .item()
            )

    def calculate_lpips(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        d = self.loss_fn_alex(tensor1.to(self.device), tensor2.to(self.device))
        return d.mean().item()

    def evaluate_batch(self, fake_tensors: torch.Tensor, real_tensors: torch.Tensor):
        if fake_tensors.shape != real_tensors.shape:
            real_tensors = torch.nn.functional.interpolate(
                real_tensors,
                size=fake_tensors.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        fake_np = self.preprocess(fake_tensors)
        real_np = self.preprocess(real_tensors)

        n_samples = fake_tensors.shape[0]
        sum_psnr = 0.0
        sum_ssim = 0.0
        sum_mse = 0.0
        sum_uqi = 0.0
        count = 0

        pbar = tqdm(range(n_samples), desc="Calculating Metrics")
        def extract_scalar(value):
            if isinstance(value, tuple):
                return value[0]
            return value

        for i in pbar:
            f_img = fake_np[i]
            r_img = real_np[i]
            gray_fake = cv2.cvtColor(f_img, cv2.COLOR_RGB2GRAY)
            gray_real = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)
            gray_fake = np.clip(gray_fake, 0, 255).astype(np.uint8)
            gray_real = np.clip(gray_real, 0, 255).astype(np.uint8)
            sum_mse += extract_scalar(sewar_mse(gray_real, gray_fake))
            sum_psnr += extract_scalar(sewar_psnr(gray_real, gray_fake))
            sum_ssim += extract_scalar(sewar_ssim(gray_real, gray_fake))
            sum_uqi += extract_scalar(sewar_uqi(gray_real, gray_fake))
            count += 1
            pbar.set_postfix({
                "PSNR": f"{sum_psnr/count:.2f}" if count else "0",
                "SSIM": f"{sum_ssim/count:.4f}" if count else "0",
                "LPIPS": "calc..."
            })
        if count == 0 or fake_tensors.numel() == 0 or real_tensors.numel() == 0:
            return {
                "PSNR(dB)": 0.0,
                "MSE": 0.0,
                "SSIM": 0.0,
                "UQI": 0.0,
                "VIF": 0.0,
                "LPIPS": 0.0,
            }

        with torch.no_grad():
            lpips_val = self.calculate_lpips(fake_tensors, real_tensors)
            vif_val = self.calculate_vif(fake_tensors, real_tensors)

        return {
            "PSNR(dB)": sum_psnr / count,
            "MSE": sum_mse / count,
            "SSIM": sum_ssim / count,
            "UQI": sum_uqi / count,
            "VIF": vif_val,
            "LPIPS": lpips_val,
        }
