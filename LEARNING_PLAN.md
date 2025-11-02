# 学习计划 (Learning Plan)

本文件是项目的唯一权威进度文档，用于统筹学习路线、追踪待办清单 (checklist)，以及保存交接日志 (handoff log)。请在完成任何子任务后同步更新本文件的复选框和日志，以便下一位协作者或 AI 助手立刻掌握状态。

## 仓库结构 (Repository Layout)

- `phases/phaseN/`：每个阶段的独立工作区，包含 `scripts/`、`outputs/`、`models/` 等子目录，以及阶段说明文件。
- `docs/`：补充参考资料、上传的论文等。
- `.venv/`：通过 `uv` 管理的 Python 虚拟环境（如需重建请参考阶段 0 指南）。

> 更新规范：完成某个阶段条目的勾选后，同步在“交接日志”新增记录；进入新阶段时先阅读对应 `phases/phaseN/README.md`，了解已有成果。

## 阶段速览 (Phase Overview)

| 阶段 (Phase) | 范围 (Scope) | 时间窗口 (Window) | 核心交付 (Deliverable) |
|--------------|--------------|-------------------|-------------------------|
| 0 | 工具与数学预备 | 1–2 天 | 本地运行 DDPM 图像示范 |
| 1 | 扩散直觉 | 第 1 周 | 可视化 + DDPM 讲解稿 |
| 2 | 采样提速 | 第 2 周 | DDPM vs DDIM 对比笔记 |
| 3 | 扩散策略 | 第 3–4 周 | Gym 滚动执行 + BC 基线 |
| 4 | 引导规划 | 第 5 周 | 轨迹引导实验报告 |
| 5 | 流匹配 | 第 6 周 | 扩散 vs 流匹配评估 |
| 6 | 系统整合 | 第 7–8 周 | 端到端机器人/车端方案 |

---

## 阶段 0 — 预备：工具与背景 (Phase 0 — Foundations & Tooling，1–2 天)

- **目标 (Goal)：** 确保环境配置与数学背景满足后续实验。
- **核心概念 (Core Concepts)：** 高斯分布 (Gaussian distribution)、KL 散度 (KL divergence)、基础神经网络结构、PyTorch 项目脚手架。
- **阅读资料 (Reading References)：**
  - `Step-by-Step Diffusion- An Elementary Tutorial.pdf` 中需要复习的章节。
  - `The Principles of Diffusion Models From Origins to Advances.pdf` 的入门部分。
- **实践清单 (Practical Checklist)：**
  - [x] 使用 `uv` 创建并管理虚拟环境 (`.venv`)。
  - [x] 安装 `torch`、`diffusers`、`pytorch-lightning`、`matplotlib`、`gradio`。
  - [x] 运行 `phases/phase0/scripts/run_ddpm_demo.py`，生成 `phases/phase0/outputs/ddpm_sample.png`。
  - [x] 导出依赖快照 `phases/phase0/outputs/environment_freeze.txt`。
  - [x] 调用 `phases/phase0/scripts/export_ddpm_samples.py`，生成 9 张样本与拼图。
  - [ ] 将代表性截图与硬件说明加入仓库（待完成）。
- **运行指南 (How to Run)：**（详见 `phases/phase0/README.md`）
  1. 创建环境：`UV_CACHE_DIR=.uv_cache uv venv .venv && source .venv/bin/activate`。
  2. 安装依赖：`UV_CACHE_DIR=.uv_cache uv pip install --python .venv/bin/python torch diffusers pytorch-lightning matplotlib gradio`。
  3. 单张采样：`.venv/bin/python phases/phase0/scripts/run_ddpm_demo.py`。
  4. 批量导出：`.venv/bin/python phases/phase0/scripts/export_ddpm_samples.py`（输出位于 `phases/phase0/outputs/ddpm_samples/`）。
  5. Gradio 演示：`.venv/bin/python phases/phase0/scripts/ddpm_gradio_app.py`，浏览器访问 `http://127.0.0.1:7860/`。
  6. 如需加速模型加载：`UV_CACHE_DIR=.uv_cache uv pip install --python .venv/bin/python accelerate`。
- **原理说明 (Principles)：**
  - `run_ddpm_demo.py`：展示 DDPM 从标准正态噪声逐步去噪的逆过程，最终还原出 CIFAR-10 分布（十类 32×32 自然图像）的样本。
  - `ddpm_gradio_app.py`：可视化采样步数 (num_inference_steps) 与随机种子 (seed) 对图像质量的影响；步数越多越接近概率流 ODE 的精确解。
  - `export_ddpm_samples.py`：批量生成并拼接样本，直观对比不同随机初始化下的多模态输出。
- **归档产出 (Outputs to Archive)：**
  - [ ] 代表性图像或动画的截图。
  - [ ] 硬件/环境说明（CPU/GPU、内存限制）。
- **进度记录 (Progress Capture)：** 全部迁移至下方“交接日志”。

---

## 阶段 1 — 打牢扩散直觉 (Phase 1 — Diffusion Intuition，第 1 周)

- **目标 (Goal)：** 理解前向加噪与反向去噪的数学与直觉。
- **核心概念 (Core Concepts)：** 噪声调度、公式 \( x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon \)、噪声预测 vs 样本预测、证据下界 (ELBO)。
- **阅读资料：**
  - `Step-by-Step Diffusion- An Elementary Tutorial.pdf` 第 2–4 节。
  - `The Principles of Diffusion Models From Origins to Advances.pdf` 第 1–3 章。
- **实践清单：**
  - [ ] 生成 1D 高斯玩具数据，绘制分布。
  - [ ] 实现正向加噪并做动画/可视化。
  - [ ] 训练最小 MLP 去噪器 (denoiser) 预测干净样本。
- **输出：**
  - [ ] 加噪可视化脚本/笔记本。
  - [ ] 去噪训练损失曲线。
  - [ ] 解释加噪、预测、采样三要素的文字稿。
- **进度记录：** 完成后将日志附加到本文件的“交接日志”。

---

## 阶段 2 — 采样效率提升 (Phase 2 — Sampling Efficiency，第 2 周)

- **目标：** 理解采样速度与生成质量的权衡。
- **核心概念：** DDPM 随机采样、DDIM 确定性采样、概率流 ODE。
- **阅读资料：**
  - `Step-by-Step Diffusion- An Elementary Tutorial.pdf` 的 DDIM 章节与速度场解释。
  - `The Principles of Diffusion Models From Origins to Advances.pdf` 第 9 章 “Fast Sampling”。
- **实践清单：**
  - [ ] 在玩具数据上实现 DDPM 与 DDIM 采样。
  - [ ] 对 1000、50、10 步进行质量比较并记录指标。
  - [ ] 编写对比两种更新公式的伪代码。
- **输出：**
  - [ ] 质量对比图或指标。
  - [ ] 标注 ODE 视角的伪代码。
  - [ ] 少步采样风险分析。

---

## 阶段 3 — 扩散策略 (Phase 3 — Diffusion Policy for Robotics，第 3–4 周)

- **目标：** 将扩散建模迁移到动作序列的生成。
- **核心概念：** 动作序列扩散、策略结构设计、滚动时域执行 (receding horizon)。
- **阅读资料：** `Diffusion Policy- Visuomotor Policy Learning via Action Diffusion.pdf` 第 3.2、4.1、4.3 节。
- **实践清单：**
  - [ ] 实现输入观测、输出未来动作序列的扩散策略。
  - [ ] 构建或选用 Gym 环境进行评估。
  - [ ] 训练行为克隆 (BC) 基线并对比。
  - [ ] 采用扩散策略进行闭环滚动（计划 N 步，执行前 k 步）。
- **输出：**
  - [ ] BC vs 扩散策略训练曲线。
  - [ ] 滚动执行轨迹或视频。
  - [ ] 多模态性、鲁棒性总结。

---

## 阶段 4 — 引导规划 (Phase 4 — Guided Trajectory Planning，第 5 周)

- **目标：** 将扩散模型作为轨迹规划器，并研究引导 (guidance) 机制。
- **核心概念：** 条件扩散、无分类器引导 (classifier-free guidance)、联合预测与规划。
- **阅读资料：** `DIFFUSION-BASED PLANNING FOR AUTONOMOUS DRIVING WITH FLEXIBLE GUIDANCE.pdf`。
- **实践清单：**
  - [ ] 构建二维障碍 + 目标的玩具场景。
  - [ ] 训练/适配扩散规划器生成可行轨迹。
  - [ ] 实现引导约束并暴露引导尺度。
  - [ ] 可视化不同引导强度下的轨迹。
- **输出：**
  - [ ] 基线 vs 引导轨迹对比图。
  - [ ] 安全性与激进度分析。
  - [ ] 引导实现伪代码。

---

## 阶段 5 — 流匹配对比 (Phase 5 — Flow Matching Alternatives，第 6 周)

- **目标：** 理解流匹配 (flow matching) 作为连续极限，与扩散规划器比较。
- **核心概念：** 概率流 ODE、流匹配公式、时空分词 (spatiotemporal tokenization)。
- **阅读资料：** `Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling.pdf` 第 3.1、3.3 节。
- **实践清单：**
  - [ ] 将阶段 4 规划器改写为流匹配训练目标。
  - [ ] 对比两者收敛速度、推理延迟、轨迹质量。
  - [ ] 记录定性差异。
- **输出：**
  - [ ] 训练/推理指标对比表。
  - [ ] 轨迹叠加图。
  - [ ] 优缺点总结。

---

## 阶段 6 — 系统整合 (Phase 6 — System Integration，第 7–8 周)

- **目标：** 将模型部署到机器人或自动驾驶系统。
- **核心概念：** 数据管线、推理服务、ROS/Isaac Sim/CARLA 集成、策略切换。
- **阅读资料：**
  - `Robot Learning- A Tutorial.pdf`。
  - `The Principles of Diffusion Models From Origins to Advances.pdf` 终章（训练技巧与未来方向）。
- **实践清单：**
  - [ ] 设计统一项目结构：
    ```
    dataset/
    model/
    rollout/
    eval/
    ```
  - [ ] 构建观测到动作序列的数据管线。
  - [ ] 实现可切换策略模式：BC、扩散策略、流匹配。
  - [ ] 在 ROS / Isaac Sim / CARLA 中验证实时性能（关注采样步数、模型蒸馏）。
- **输出：**
  - [ ] 仓库级 README（启动流程）。
  - [ ] 策略指标：延迟、成功率等。
  - [ ] 部署限制与未来扩展。

---

## 交接日志 (Handoff Log)

> 记录格式：`## YYYY-MM-DD — 阶段 X 状态 (Phase X Status)`，并保持中文 + 英文括注的风格。

## 2025-11-02 — 阶段 0 状态 (Phase 0 Status)
- **已完成 (Completed)：** 使用 uv 创建 `.venv`；安装 `torch`、`diffusers`、`pytorch-lightning`、`matplotlib`、`gradio`；运行 `phases/phase0/scripts/run_ddpm_demo.py`，生成 `phases/phase0/outputs/ddpm_sample.png`；导出依赖 `phases/phase0/outputs/environment_freeze.txt`；编写 `phases/phase0/scripts/ddpm_gradio_app.py` 并完成本地验证；批量导出脚本 `phases/phase0/scripts/export_ddpm_samples.py` 产出 9 张样本及拼图 `phases/phase0/outputs/ddpm_samples/ddpm_samples_grid.png`。
- **进行中 (In Progress)：** 无。
- **阻塞 / 问题 (Blockers / Questions)：** `uv` 默认缓存目录无写权限，需设置 `UV_CACHE_DIR=/workspaces/Learn_Diffusion_Model/.uv_cache`；初始缺少 `uv`，通过 `python -m pip install uv` 并申请网络访问解决；`.venv/bin/python` 未预装 `pip`，改用 `uv pip freeze` 导出依赖；运行 DDPM demo 时提示缺少 `accelerate`，后续可安装以降低 CPU 内存占用。
- **下一步建议 (Next Suggested Actions)：** 进入阶段 1，阅读 `Step-by-Step Diffusion` 第 2–4 节并实现正向加噪/去噪可视化；如需频繁运行 diffusers，可安装 `accelerate`；Gradio 演示运行命令：`.venv/bin/python phases/phase0/scripts/ddpm_gradio_app.py`（默认端口 `http://127.0.0.1:7860/`）。

---

## 附录 (Appendix)

- **交接模板 (Handoff Template)：**
  ```
  ## YYYY-MM-DD — 阶段 X 状态 (Phase X Status)
  - 已完成 (Completed)：...
  - 进行中 (In Progress)：...
  - 阻塞 / 问题 (Blockers / Questions)：...
  - 下一步建议 (Next Suggested Actions)：...
  ```
- **更新流程 (Update Flow)：**
  1. 执行任务 → 勾选对应 checklist。
  2. 在“交接日志”追加记录。
  3. 如有新脚本/输出，放入对应 `phases/phaseN/` 目录并在 `README.md` 说明。
