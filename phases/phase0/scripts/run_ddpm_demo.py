"""
Minimal DDPM (Denoising Diffusion Probabilistic Model) sampling demo.

Downloads a pre-trained DDPM model from Hugging Face and generates a single
image sample, saving it under outputs/phase0/ddpm_sample.png.
"""

from pathlib import Path

from diffusers import DDPMPipeline


PHASE_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = PHASE_ROOT / "models"
OUTPUT_PATH = PHASE_ROOT / "outputs" / "ddpm_sample.png"


def main() -> None:
    # Use a small CIFAR-10 model to keep download and sampling lightweight.
    pipeline = DDPMPipeline.from_pretrained(
        "google/ddpm-cifar10-32",
        cache_dir=MODEL_CACHE,
    )
    pipeline = pipeline.to("cpu")

    image = pipeline(num_inference_steps=50).images[0]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH)

    print(f"Saved DDPM sample to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
