"""
DDPM 可视化演示 (DDPM Visualization Demo)。

通过 Gradio 提供交互界面，加载预训练的 DDPMPipeline 并生成 CIFAR-10 图像样本，
方便在浏览器中调整采样步数 (num_inference_steps) 与随机种子 (seed)。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import gradio as gr
import torch
from diffusers import DDPMPipeline

MODEL_ID = "google/ddpm-cifar10-32"
PHASE_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = PHASE_ROOT / "models"
_pipeline: Optional[DDPMPipeline] = None


def _load_pipeline() -> DDPMPipeline:
    """延迟加载预训练管线 (lazy load pipeline)，避免重复下载 (avoid redundant downloads)。"""
    global _pipeline
    if _pipeline is None:
        MODEL_CACHE.mkdir(exist_ok=True)
        _pipeline = DDPMPipeline.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)
        _pipeline = _pipeline.to("cpu")
    return _pipeline


def generate_images(num_steps: int, num_images: int, seed: int) -> List:
    """调用管线生成图像 (generate images) 并以 PIL 列表返回。"""
    pipe = _load_pipeline()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        batch_size=num_images,
        num_inference_steps=num_steps,
        generator=generator,
    )
    return result.images


def build_interface() -> gr.Blocks:
    """构建 Gradio 界面 (build Gradio interface)。"""
    with gr.Blocks(title="DDPM Demo") as demo:
        gr.Markdown(
            """
            ## DDPM 图像生成演示 (DDPM Image Generation Demo)

            1. **输出含义 (What You See)**：这些 32×32 的彩色图片来自 DDPM 在 CIFAR-10 数据集上学习到的分布，
               因此会呈现飞机、汽车、卡车、鸟类等十类自然图像的典型纹理与颜色。
            2. **采样流程 (Sampling Pipeline)**：界面会调用 `DDPMPipeline`，从纯噪声 \(x_T\) 开始，
               逐步执行反向扩散步骤 (reverse diffusion steps) 去掉预测噪声 \\( \epsilon_\\theta(x_t, t) \\)，
               最终得到类似 CIFAR-10 的样本。
            3. **采样步数 (num_inference_steps)**：步数越高，近似真实概率流 ODE 的能力越好，图像更清晰；
               步数过少则会出现残留噪声和模糊结构。
            4. **随机种子 (seed)**：固定种子可重复获得相同的噪声初始化，方便比较步数或模型设置。
            5. **生成数量 (num_images)**：一次生成多张样本，以观察多模态输出与同类之间的差异。

            点击 **Generate** 后，下方画廊会展示生成结果以及不同参数组合的效果差异。
            """
        )
        with gr.Row():
            num_steps = gr.Slider(
                minimum=10,
                maximum=200,
                value=50,
                step=10,
                label="采样步数 (num_inference_steps)",
            )
            num_images = gr.Slider(
                minimum=1,
                maximum=4,
                value=1,
                step=1,
                label="生成数量 (num_images)",
            )
            seed = gr.Number(
                value=42,
                precision=0,
                label="随机种子 (seed)",
            )
        generate_button = gr.Button("Generate")
        gallery = gr.Gallery(
            label="生成结果 (Generated Samples)",
            show_label=True,
            columns=2,
            height=400,
        )

        generate_button.click(
            fn=lambda steps, images, seed_value: generate_images(
                int(steps), int(images), int(seed_value)
            ),
            inputs=[num_steps, num_images, seed],
            outputs=[gallery],
        )
    return demo


def main() -> None:
    """启动 Gradio 应用 (launch Gradio app)。"""
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
