# Phase 0 概览 (Phase 0 Overview)

本阶段目标是搭建基础环境并熟悉最小扩散模型 (DDPM) 的推理流程，同时为后续阶段准备工具链。

## 目录说明 (Directory Map)

- `scripts/`：阶段 0 使用的所有脚本  
  - `run_ddpm_demo.py`：单张 CIFAR-10 样本生成演示。  
  - `ddpm_gradio_app.py`：Gradio 可视化界面，调整采样步数与随机种子。  
  - `export_ddpm_samples.py`：批量导出 9 张样本并生成 3×3 拼图。
- `outputs/`：默认生成的产物与依赖快照  
  - `ddpm_sample.png`：单次采样结果示例。  
  - `ddpm_samples/`：批量样本与 `ddpm_samples_grid.png` 拼图。  
  - `environment_freeze.txt`：通过 `uv pip freeze` 导出的环境依赖。
- `models/`：Hugging Face 预训练权重缓存目录（首次运行脚本会自动填充）。

## 快速开始 (Quick Start)

```bash
# 激活虚拟环境
source .venv/bin/activate

# 单张采样
python phases/phase0/scripts/run_ddpm_demo.py

# 批量导出并生成拼图
python phases/phase0/scripts/export_ddpm_samples.py

# 启动 Gradio 可视化
python phases/phase0/scripts/ddpm_gradio_app.py
```

## 结果解析 (Result Notes)

- 所有生成图片来自 CIFAR-10 数据分布（飞机、汽车、动物等十类 32×32 彩色图像）。  
- 采样步数 (num_inference_steps) 越多，重建出的细节越清晰；步数不足会残留噪声。  
- Gradio 界面允许一次生成多张样本，方便观察同类之间的多模态差异。

更新或新增脚本、输出时，请同步修改本 README 与根目录 `LEARNING_PLAN.md` 中的阶段 0 checklist。
