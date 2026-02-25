#!/usr/bin/env python3
"""
Keep only object pixels from an RGB image using a mask.
Background pixels are set to a dark value (default 0).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask background to dark and retain only object pixels."
    )
    parser.add_argument(
        "--rgb",
        required=True,
        type=str,
        help="Path to input RGB image.",
    )
    parser.add_argument(
        "--mask",
        required=True,
        type=str,
        help="Path to input mask image (non-zero means object).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Path to output image. Default: <rgb_stem>_obj.png next to input rgb.",
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=0,
        help="Mask threshold. Pixels > threshold are treated as object.",
    )
    parser.add_argument(
        "--bg_value",
        type=int,
        default=0,
        help="Background value [0,255]. Default 0 (black).",
    )
    return parser.parse_args()


def load_image(path: str, read_flag: int):
    img = cv2.imread(path, read_flag)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def main():
    args = parse_args()

    rgb_path = Path(args.rgb).expanduser().resolve()
    mask_path = Path(args.mask).expanduser().resolve()
    if not rgb_path.exists():
        raise FileNotFoundError(str(rgb_path))
    if not mask_path.exists():
        raise FileNotFoundError(str(mask_path))

    rgb = load_image(str(rgb_path), cv2.IMREAD_UNCHANGED)
    mask = load_image(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(
            mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    fg = mask > int(args.mask_threshold)
    bg_value = int(np.clip(args.bg_value, 0, 255))

    out = np.full_like(rgb, bg_value)
    out[fg] = rgb[fg]

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = rgb_path.with_name(f"{rgb_path.stem}_obj.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(out_path), out)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {out_path}")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
