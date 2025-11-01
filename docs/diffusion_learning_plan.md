# 扩散机器人学习计划 (Diffusion Robotics Learning Program)

本文档以分阶段路线图的形式整理学习路径，帮助学习者从扩散基础 (diffusion basics) 一步步走到能够训练与部署扩散策略 (diffusion policy) 与规划器 (diffusion-based planner)，并最终整合到机器人与自动驾驶系统中。每个阶段都同时安排理论理解 (theory)、实践实现 (practice) 与产出物 (deliverable)，便于后续协作者或 AI 助手快速接力。

## 项目概要 (Program Brief)

- **整体目标 (Primary Objective)：** 掌握训练、评估与部署扩散策略 (diffusion policy) 与流匹配策略 (flow-matching policy) 的能力，覆盖视觉控制 (visuomotor control) 与自动驾驶 (autonomous driving) 场景。
- **预计周期 (Estimated Cadence)：** 6–8 周；每周建议投入 8–10 小时。
- **知识基础 (Background Focus)：** Python、PyTorch、概率建模 (probabilistic modeling)、深度学习 (deep learning)、机器人控制系统 (robotics control stack)。
- **进度记录 (Progress Tracking)：** 每完成一个阶段，就更新核对清单、把代码/笔记/图表等产出保存进仓库，并在交接日志 (handoff log) 中记录阻塞点与下一步。

## 阶段速览 (Phase Overview)

| 阶段 (Phase) | 范围 (Scope) | 时间窗口 (Window) | 核心交付 (Deliverable) |
|--------------|--------------|-------------------|-------------------------|
| 0 | 工具与数学预备 | 1–2 天 | 本地运行 DDPM 图像示范 (DDPM image demo) |
| 1 | 扩散直觉 | 第 1 周 | 可视化 + DDPM 讲解稿 |
| 2 | 采样提速 | 第 2 周 | DDPM vs DDIM 对比笔记 |
| 3 | 扩散策略 | 第 3–4 周 | Gym 环境滚动执行 + BC 基线 |
| 4 | 引导规划 | 第 5 周 | 轨迹引导实验报告 |
| 5 | 流匹配 | 第 6 周 | 扩散 vs 流匹配评估 |
| 6 | 系统整合 | 第 7–8 周 | 端到端机器人/车端方案 |

## 阶段 0 — 预备：工具与背景 (Phase 0 — Foundations & Tooling，1–2 天)

- **目标 (Goal)：** 确保环境配置与数学背景满足后续实验。
- **核心概念 (Core Concepts)：** 高斯分布 (Gaussian distribution)、KL 散度 (KL divergence)、基础神经网络结构 (MLP/CNN/Transformer)、PyTorch 项目脚手架。
- **阅读资料 (Reading References)：**
  - `Step-by-Step Diffusion- An Elementary Tutorial.pdf` 中需要复习的章节。
  - `The Principles of Diffusion Models From Origins to Advances.pdf` 中的入门部分。
- **实践清单 (Practical Checklist)：**
  - [ ] 确认 Python ≥3.9、PyTorch、`diffusers`、`pytorch-lightning`、`matplotlib` 等已安装。
  - [ ] 在本地运行 Hugging Face 的 `DDPMPipeline` 示例，并保存输出图片。
  - [ ] 在交接日志 (handoff log) 记录环境版本与依赖 (`pip freeze` 摘要)。
- **归档产出 (Outputs to Archive)：**
  - [ ] 生成样本的截图或图表。
  - [ ] 关于硬件条件 (GPU/内存) 的简要说明。
- **进度记录 (Progress Capture)：** 汇总环境搭建过程中遇到的阻塞与解决方法，方便复用。

## 阶段 1 — 打牢扩散直觉 (Phase 1 — Diffusion Intuition，第 1 周)

- **目标 (Goal)：** 理解正向加噪 (forward diffusion) 与反向去噪 (reverse denoising) 的数学与直觉。
- **核心概念 (Core Concepts)：** 噪声调度 (noise schedule)、公式 \( x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon \)、噪声预测目标 (noise prediction objective) 与样本预测目标 (sample prediction objective)、证据下界 (ELBO)。
- **阅读资料 (Reading References)：**
  - `Step-by-Step Diffusion- An Elementary Tutorial.pdf` 第 2–4 节。
  - `The Principles of Diffusion Models From Origins to Advances.pdf` 第 1–3 章。
- **实践清单 (Practical Checklist)：**
  - [ ] 使用 NumPy/PyTorch 生成 1D 高斯玩具数据。
  - [ ] 实现正向加噪过程，并生成清晰到模糊的动画。
  - [ ] 训练最小 MLP 去噪器 (denoiser) 以从噪声预测干净样本。
- **归档产出 (Outputs to Archive)：**
  - [ ] 展示正向加噪可视化的脚本或笔记本。
  - [ ] 去噪训练曲线 (loss 随训练步数变化)。
  - [ ] 说明加噪 (noising)、预测 (prediction)、采样 (sampling) 三大环节的短文。
- **进度记录 (Progress Capture)：** 在交接日志中记录理解要点与未解决疑问。

## 阶段 2 — 采样效率提升 (Phase 2 — Sampling Efficiency，第 2 周)

- **目标 (Goal)：** 理解采样速度与生成质量的权衡。
- **核心概念 (Core Concepts)：** DDPM 随机采样 (stochastic sampling)、DDIM 确定性采样 (deterministic sampling)、概率流常微分方程 (probability flow ODE)。
- **阅读资料 (Reading References)：**
  - `Step-by-Step Diffusion- An Elementary Tutorial.pdf` 中的 DDIM 章节与速度场 (velocity field) 解释。
  - `The Principles of Diffusion Models From Origins to Advances.pdf` 第 9 章 “Fast Sampling”。
- **实践清单 (Practical Checklist)：**
  - [ ] 在玩具数据集上分别实现 DDPM 与 DDIM 采样流程。
  - [ ] 调整采样步数：1000 → 50 → 10，观察质量变化并记录指标。
  - [ ] 写出比较 DDPM 与 DDIM 更新公式的伪代码 (pseudocode)。
- **归档产出 (Outputs to Archive)：**
  - [ ] 不同步数下的结果图或代理指标 (例如简化版 FID)。
  - [ ] 标注 ODE 视角要点的伪代码片段。
  - [ ] 关于少步采样风险与适用场景的反思记录。
- **进度记录 (Progress Capture)：** 记录实验中调参经验与采样异常案例。

## 阶段 3 — 从视觉到动作的扩散策略 (Phase 3 — Diffusion Policy for Robotics，第 3–4 周)

- **目标 (Goal)：** 将扩散建模思想迁移到机器人动作序列生成。
- **核心概念 (Core Concepts)：** 动作序列扩散 (action sequence diffusion)、策略结构设计 (policy architecture)、滚动时域执行 (receding horizon execution)。
- **阅读资料 (Reading References)：**
  - `Diffusion Policy- Visuomotor Policy Learning via Action Diffusion.pdf` 第 3.2、4.1、4.3 节。
- **实践清单 (Practical Checklist)：**
  - [ ] 实现一个简化扩散策略，输入观测向量输出未来动作序列。
  - [ ] 构建或选择一个 Gym 环境，用于评估策略效果。
  - [ ] 训练行为克隆 (behavior cloning, BC) 基线进行对比。
  - [ ] 采用扩散策略进行闭环滚动：每次采样未来 N 步，仅执行前 k 步。
- **归档产出 (Outputs to Archive)：**
  - [ ] BC 与扩散策略的训练曲线。
  - [ ] 滚动执行的视频或状态轨迹对比图。
  - [ ] 总结扩散策略在平滑性 (smoothness)、鲁棒性 (robustness)、多模态性 (multimodality) 上的优势。
- **进度记录 (Progress Capture)：** 记录策略结构、数据集配置与尚待优化的问题。

## 阶段 4 — 自动驾驶中的引导扩散规划 (Phase 4 — Guided Trajectory Planning，第 5 周)

- **目标 (Goal)：** 理解扩散模型作为轨迹规划器 (trajectory planner) 的角色，并探索引导 (guidance) 策略。
- **核心概念 (Core Concepts)：** 条件扩散 (conditional diffusion)、无分类器引导 (classifier-free guidance)、联合预测与规划 (joint prediction and planning)。
- **阅读资料 (Reading References)：**
  - `DIFFUSION-BASED PLANNING FOR AUTONOMOUS DRIVING WITH FLEXIBLE GUIDANCE.pdf`。
- **实践清单 (Practical Checklist)：**
  - [ ] 构建二维轨迹场景：含障碍物与目标点的玩具环境。
  - [ ] 训练或微调扩散模型以生成可行轨迹。
  - [ ] 加入引导约束（如安全距离、安全速度），并开放引导尺度 (guidance scale) 参数。
  - [ ] 可视化不同引导强度下的轨迹变化。
- **归档产出 (Outputs to Archive)：**
  - [ ] 基线与引导轨迹的比较图。
  - [ ] 关于安全性与激进度 (aggressiveness) 权衡的文字分析。
  - [ ] 引导实现的伪代码或代码片段，方便复用。
- **进度记录 (Progress Capture)：** 记录约束设计、失败案例（碰撞、震荡）与解决思路。

## 阶段 5 — 探索流匹配替代方案 (Phase 5 — Flow Matching Alternatives，第 6 周)

- **目标 (Goal)：** 掌握流匹配 (flow matching) 作为连续极限方法，并与扩散规划器对比。
- **核心概念 (Core Concepts)：** 概率流 ODE (probability flow ODE)、流匹配公式 (flow matching formulation)、时空分词 (spatiotemporal tokenization)。
- **阅读资料 (Reading References)：**
  - `Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling.pdf` 第 3.1、3.3 节。
- **实践清单 (Practical Checklist)：**
  - [ ] 将阶段 4 的扩散规划器改写为流匹配训练目标。
  - [ ] 对比两种方法的收敛速度、推理延迟与轨迹质量。
  - [ ] 记录定性差异（例如轨迹形态、稳定性）。
- **归档产出 (Outputs to Archive)：**
  - [ ] 训练轮次、推理时间、轨迹指标的对比表。
  - [ ] 典型轨迹的叠加可视化。
  - [ ] 梳理两者优缺点的报告。
- **进度记录 (Progress Capture)：** 提炼潜在的混合方案（如蒸馏 distillation、课程学习 curriculum）。

## 阶段 6 — 工程化整合与部署 (Phase 6 — System Integration，第 7–8 周)

- **目标 (Goal)：** 将模型集成进真实机器人或自动驾驶系统的工程栈。
- **核心概念 (Core Concepts)：** 数据管线 (data pipeline)、推理服务 (inference pipeline)、ROS/Isaac Sim/CARLA 集成、策略切换 (policy switching)。
- **阅读资料 (Reading References)：**
  - `Robot Learning- A Tutorial.pdf`。
  - `The Principles of Diffusion Models From Origins to Advances.pdf` 结尾关于训练技巧与未来方向的章节。
- **实践清单 (Practical Checklist)：**
  - [ ] 设计项目结构：
    ```
    dataset/
    model/
    rollout/
    eval/
    ```
  - [ ] 构建数据读取与预处理脚本，将观测转为动作序列。
  - [ ] 实现可切换策略模式：BC、扩散策略、流匹配策略。
  - [ ] 在 ROS / Isaac Sim / CARLA 中验证实时性能，关注采样步数与模型蒸馏 (model distillation)。
- **归档产出 (Outputs to Archive)：**
  - [ ] 仓库级 README，说明启动流程与依赖。
  - [ ] 各策略模式的关键指标：延迟 (latency)、成功率 (success rate) 等。
  - [ ] 部署限制与后续扩展想法。
- **进度记录 (Progress Capture)：** 维护整合过程中的阻塞清单（模拟器配置、传感器延迟等）及对应解决方案。

## 交接日志模板 (Handoff Log Template)

为确保不同协作者或 AI 助手可以无缝衔接，请在 `docs/handoff_log.md`（若不存在则创建）中使用以下格式追加记录：

```
## YYYY-MM-DD — 阶段 X 状态 (Phase X Status)
- 已完成 (Completed)：...
- 进行中 (In Progress)：...
- 阻塞 / 问题 (Blockers / Questions)：...
- 下一步建议 (Next Suggested Actions)：...
```

请务必在日志中附上相关代码、笔记本或图表的路径，确保学习过程可复现、易协作。
