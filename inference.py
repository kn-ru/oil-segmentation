"""
Sliding-window inference для DualPolMAFUformerMIL с Test-Time Augmentation.

Стратегия:
  - Окно 512x512, stride 384
  - Перекрывающиеся области усредняются
  - TTA: flips (h, v, hv) + original → 4x averaging
  - Классификация: bag classifier на tile embeddings
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import rasterio

from config import Config
from model import DualPolMAFUformerMIL
from dataset import SARNormalizer
from utils import set_seed

import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


CLASS_NAMES = ["oil", "lookalike", "no_oil"]


def sliding_window_coords(H: int, W: int, window: int = 512,
                          stride: int = 384):
    coords = set()
    for y in range(0, H - window + 1, stride):
        for x in range(0, W - window + 1, stride):
            coords.add((y, x))

    # Крайние окна
    if H >= window and W >= window:
        for x in range(0, W - window + 1, stride):
            coords.add((H - window, x))
        for y in range(0, H - window + 1, stride):
            coords.add((y, W - window))
        coords.add((H - window, W - window))

    return list(coords)


# ──────────────────── TTA transforms ───────────────────────────────────

def _flip_batch(vv: torch.Tensor, vh: torch.Tensor, mode: str):
    """Apply flip augmentation."""
    if mode == "none":
        return vv, vh
    elif mode == "h":
        return vv.flip(-1), vh.flip(-1)
    elif mode == "v":
        return vv.flip(-2), vh.flip(-2)
    elif mode == "hv":
        return vv.flip(-1).flip(-2), vh.flip(-1).flip(-2)
    raise ValueError(f"Unknown TTA mode: {mode}")


def _unflip(mask: torch.Tensor, mode: str):
    """Reverse flip on mask predictions."""
    if mode == "none":
        return mask
    elif mode == "h":
        return mask.flip(-1)
    elif mode == "v":
        return mask.flip(-2)
    elif mode == "hv":
        return mask.flip(-1).flip(-2)
    raise ValueError(f"Unknown TTA mode: {mode}")


# ──────────────────── Inference ────────────────────────────────────────

@torch.no_grad()
def predict_image(
    model: DualPolMAFUformerMIL,
    vv: np.ndarray,
    vh: np.ndarray,
    normalizer: SARNormalizer,
    device: torch.device,
    window: int = 512,
    stride: int = 384,
    batch_size: int = 8,
    tta: bool = True,
) -> dict:
    """
    Предсказание для одного полного изображения.

    Args:
        model: обученная модель
        vv, vh: (H, W) raw SAR in dB
        normalizer: обученный нормализатор
        device: cuda/cpu
        window, stride: параметры sliding window
        batch_size: тайлов за раз
        tta: использовать test-time augmentation (4x flips)
    """
    model.eval()
    H, W = vv.shape
    vv_norm, vh_norm = normalizer.transform(vv, vh)

    coords = sliding_window_coords(H, W, window, stride)
    tta_modes = ["none", "h", "v", "hv"] if tta else ["none"]

    mask_sum = np.zeros((H, W), dtype=np.float64)
    mask_count = np.zeros((H, W), dtype=np.float64)
    all_embeddings = []

    for tta_mode in tta_modes:
        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            vv_batch = []
            vh_batch = []

            for y, x in batch_coords:
                vv_batch.append(vv_norm[y:y + window, x:x + window])
                vh_batch.append(vh_norm[y:y + window, x:x + window])

            vv_t = torch.from_numpy(np.array(vv_batch)[:, None]).to(device)
            vh_t = torch.from_numpy(np.array(vh_batch)[:, None]).to(device)

            # Apply TTA flip
            vv_aug, vh_aug = _flip_batch(vv_t, vh_t, tta_mode)

            tile_out = model.forward_tile(vv_aug, vh_aug)
            mask_logits = tile_out["mask_logits"]

            # Unflip mask predictions
            mask_logits = _unflip(mask_logits, tta_mode)
            mask_probs = torch.sigmoid(mask_logits).squeeze(1).cpu().numpy()

            if tta_mode == "none":
                all_embeddings.append(tile_out["tile_embedding"].cpu())

            for j, (y, x) in enumerate(batch_coords):
                mask_sum[y:y + window, x:x + window] += mask_probs[j]
                mask_count[y:y + window, x:x + window] += 1.0

    # Average
    mask_prob = mask_sum / np.maximum(mask_count, 1.0)

    # Classification
    all_embs = torch.cat(all_embeddings, dim=0).unsqueeze(0).to(device)
    class_logits = model.bag_classifier(all_embs).squeeze(0).cpu()
    class_pred = class_logits.argmax().item()

    return {
        "mask_prob": mask_prob.astype(np.float32),
        "class_logits": class_logits.numpy(),
        "class_pred": class_pred,
        "class_name": CLASS_NAMES[class_pred],
    }


def predict_file(model, image_path, normalizer, device,
                 window=512, stride=384, tta=True):
    with rasterio.open(image_path) as src:
        vv = src.read(1)
        vh = src.read(2)
        profile = src.profile

    result = predict_image(model, vv, vh, normalizer, device,
                           window, stride, tta=tta)
    result["profile"] = profile
    return result


def save_prediction(mask_prob, output_path, profile=None):
    if profile:
        profile.update(count=1, dtype="float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask_prob, 1)
    else:
        import tifffile
        tifffile.imwrite(output_path, mask_prob)


def main():
    parser = argparse.ArgumentParser(description="Inference DualPolMAFUformerMIL")
    parser.add_argument("input", help="Path to input TIFF or directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable test-time augmentation")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualPolMAFUformerMIL.from_config(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema" in state:
        model.load_state_dict(state["ema"])
    else:
        model.load_state_dict(state["model"])
    model.eval()

    norm_path = os.path.join(os.path.dirname(args.checkpoint), "normalizer.json")
    normalizer = SARNormalizer().load(norm_path)

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input) if f.endswith(".tif")
        ])

    use_tta = not args.no_tta
    print(f"Processing {len(files)} files (TTA={'on' if use_tta else 'off'})...")

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"  {fname}...", end=" ")

        result = predict_file(model, fpath, normalizer, device,
                              stride=args.stride, tta=use_tta)

        mask_path = os.path.join(args.output_dir, f"mask_{fname}")
        save_prediction(result["mask_prob"], mask_path, result.get("profile"))

        binary = (result["mask_prob"] > args.threshold).astype(np.uint8)
        binary_path = os.path.join(args.output_dir, f"binary_{fname}")
        save_prediction(binary.astype(np.float32), binary_path, result.get("profile"))

        print(f"class={result['class_name']}, "
              f"oil_area={result['mask_prob'].mean():.4f}")

    print("Done!")


if __name__ == "__main__":
    main()
