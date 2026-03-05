

---

# 🚀 AI4S Public Course: Lesson 1 Construction Task

**Context:** 为非 AI 背景的研究生/程序员构建一个关于“数字全息图：特征表征”的 Jupyter Notebook 实操教程。
**Core Theme:** 流体动力学中的降维与重构 (Dimensionality Reduction & Reconstruction in Fluid Dynamics)link:https://www.csc.kth.se/~weinkauf/datasets/Cylinder2D.7z.

---

## 📋 Phase 1: Environment & Data Setup

* [ ] **Dependencies:** 生成 `requirements.txt`。需包含 `torch`, `numpy`, `matplotlib`, `scipy`, `ipywidgets`, `scikit-learn`。
* [ ] **Synthetic Data Generation:** 编写一个脚本 `generate_data.py`，模拟或加载 **2D Cylinder Flow (Vorticity Field)**。
* **Specs:** 1000 个快照, 尺寸 $64 \times 128$。
* **Normalization:** 将数据缩放到 $[-1, 1]$ 区间。
* **Output:** 保存为 `flow_data.npy`。



## 📋 Phase 2: Model Architecture (PyTorch)

* [ ] **Encoder Design:** 构建卷积神经网络。
* 3层卷积层，使用 `stride=2` 进行下采样。
* 最后一层映射到 `latent_dim=16` 的全连接层。


* [ ] **Decoder Design:** 构建对称的转置卷积网络。
* 使用 `ConvTranspose2d` 将 16 维向量还原回 $64 \times 128$ 图像。
* 最后一层使用 `Tanh` 激活函数。


* [ ] **Loss Function:** 使用 MSE Loss，并预留一个可选的“物理一致性约束” (Physics-informed) 接口。

## 📋 Phase 3: Notebook Structure (The "Cool" Part)

* [ ] **Section 1: Visualization:** 编写代码展示原始流场的卡门涡街 (Von Kármán vortex street) 动画。
* [ ] **Section 2: Training Loop:** 实现一个简洁的训练循环，带进度条（`tqdm`），展示 Loss 下降过程。
* [ ] **Section 3: Latent Space Interpolation (炫酷任务):**
* 选取两个不同时刻的流场 $z_1, z_2$。
* 在它们之间进行线性插值 $\alpha z_1 + (1-\alpha)z_2$。
* 渲染生成的中间流场，展示 AI 如何理解流体演化。


* [ ] **Section 4: Interactive Dashboard:**
* 使用 `ipywidgets` 构建滑块界面。
* 用户调节前 4 个主分量，实时显示解码后的流场图像。



## 📋 Phase 4: Educational Content (Markdown Cells)

* [ ] **Analogy:** 在单元格中加入类比说明：将卷积核比作“物理特征过滤器”，将 Latent Space 比作“物理系统的指纹”。
* [ ] **Challenge Task:** 为学生留一个 `TODO`：尝试改变 `latent_dim` 从 16 到 2，观察重构质量的变化（演示信息压缩的极限）。

---

## 🛠 Agent Execution Instructions

1. **Code Quality:** 所有 Python 代码需遵循 PEP8 规范。
2. **Comments:** 关键物理量（如涡量、雷诺数）需有中文注释。
3. **Visualization:** Matplotlib 绘图需使用 `RdBu_r` 色图，以符合流体力学审美。
4. **No-Headless:** 确保生成的代码在本地 Jupyter 环境下可交互。

---

### 💡 给 Shikai 老师的小贴士：

如果你使用的 Agent 支持调用工具，你可以让它先执行 `generate_data.py`。如果没有真实流体数据，可以让 Agent 用 `scipy.special.jn`（贝塞尔函数）或者多个高斯核叠加来模拟一个类似的“伪流体”涡旋场，视觉效果同样很棒且无需外部依赖。

**需要我帮你针对某个具体的 Agent（如 AutoGPT 或 OpenInterpreter）优化提示词吗？**