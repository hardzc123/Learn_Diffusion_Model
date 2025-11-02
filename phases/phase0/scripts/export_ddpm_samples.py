"""
批量生成 DDPM (Denoising Diffusion Probabilistic Model) 样本并保存。

默认输出 3x3 网格图和单张图片文件，便于快速浏览 CIFAR-10 风格的生成结果。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image

from diffusers import DDPMPipeline

PHASE_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = PHASE_ROOT / "models"
DEFAULT_OUTPUT_DIR = PHASE_ROOT / "outputs" / "ddpm_samples"


def load_pipeline(model_id: str, cache_dir: Path) -> DDPMPipeline:
    """加载指定模型 (load model) 并缓存到给定目录。"""
    cache_dir.mkdir(exist_ok=True)
    pipeline = DDPMPipeline.from_pretrained(model_id, cache_dir=cache_dir)
    return pipeline.to("cpu")


def generate_images(pipeline: DDPMPipeline, num_images: int, num_steps: int, seed: int) -> List[Image.Image]:
    """使用给定管线生成指定数量的图像。"""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipeline(
        batch_size=num_images,
        num_inference_steps=num_steps,
        generator=generator,
    )
    return result.images


def save_individual(images: List[Image.Image], output_dir: Path) -> None:
    """保存单张图片，文件名带索引。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(images):
        img.save(output_dir / f"ddpm_sample_{idx:02d}.png")


def save_grid(images: List[Image.Image], grid_rows: int, grid_cols: int, output_path: Path) -> None:
    """按照行列拼接成网格图。"""
    if len(images) == 0:
        raise ValueError("No images provided for grid export.")
    width, height = images[0].size
    grid = Image.new("RGB", (grid_cols * width, grid_rows * height))
    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        if row >= grid_rows:
            break
        grid.paste(img, (col * width, row * height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DDPM samples and grid image.")
    parser.add_argument("--model-id", default="google/ddpm-cifar10-32", help="预训练模型 ID (pretrained model id).")
    parser.add_argument("--num-images", type=int, default=9, help="生成图片数量 (total images to generate).")
    parser.add_argument("--num-steps", type=int, default=50, help="采样步数 (num_inference_steps).")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (seed) for reproducibility.")
    parser.add_argument("--grid-rows", type=int, default=3, help="网格行数 (grid rows).")
    parser.add_argument("--grid-cols", type=int, default=3, help="网格列数 (grid cols).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录 (output directory).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline(args.model_id, MODEL_CACHE)

    images = generate_images(pipeline, args.num_images, args.num_steps, args.seed)
    save_individual(images, args.output_dir)

    grid_path = args.output_dir / "ddpm_samples_grid.png"
    save_grid(images, args.grid_rows, args.grid_cols, grid_path)

    print(f"Saved {len(images)} images to {args.output_dir.resolve()}")
    print(f"Saved grid image to {grid_path.resolve()}")


if __name__ == "__main__":
    main()
