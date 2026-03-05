
你可以直接将以下内容复制到文件中，或者直接喂给 Agent 的 Prompt。这个版本增强了**工程落地性**，确保 Agent 不仅仅是写出代码，还能生成一个可直接运行的、美观的教学 Notebook。

---

# 🛠️ AI4S 公开课 L1 自动构建任务清单 (To-Do List)

## 🎯 任务目标

构建一个名为 `AI4S_Lesson1_Fluid_AE.ipynb` 的 Jupyter Notebook。
该 Notebook 需通过**自编码器 (Autoencoder)** 展示流体物理场（卡门涡街）的低维表征学习。
**目标受众：** 非 AI 背景的研究生、程序员（需要解释清楚物理含义）。

---

## 📅 阶段一：环境与模拟数据生成 (Data Engine)

* [ ] **生成 `requirements.txt`：**
* 必须包含：`torch`, `numpy`, `matplotlib`, `scipy`, `ipywidgets`, `tqdm`.


* [ ] **编写数据生成函数 `generate_vortex_data()`：**
* **逻辑：** 若无外部数据集，使用 `scipy` 生成伪流体场。利用多个高斯旋涡（Gaussian Vortices）的叠加与随时间的平移，模拟“卡门涡街”的脱落过程。
* **规格：** 生成 1200 帧（1000 训练, 200 测试），每帧尺寸 $64 \times 128$。
* **归一化：** 数据需映射至 $[-1, 1]$ 并保存为 `flow_field.npy`。



## 📅 阶段二：模型架构设计 (Neural Architecture)

* [ ] **构建 `FlowAE` 类 (PyTorch)：**
* **Encoder：** 3层 `Conv2d` (通道数 1->16->32->64)，配合 `Stride=2` 实现下采样，最后接 `Linear` 层压入 `latent_dim=16`。
* **Decoder：** 对称使用 `ConvTranspose2d`。最后一层使用 `Tanh` 激活以匹配数据分布。


* [ ] **Loss Function：** 使用简单的 `MSELoss`。
* [ ] **Optimizer：** 使用 `Adam` (lr=1e-3)。

## 📅 阶段三：Notebook 结构编排 (Interactive Tutorial)

* [ ] **模块 1：导论与可视化：**
* Markdown 单元格：解释“数字全息图”概念——用 16 个数字代表 8192 个网格点。
* 代码：使用 `plt.imshow(cmap='RdBu_r')` 渲染原始流场动画。


* [ ] **模块 2：训练演示：**
* 编写一个简洁的训练循环。
* 每 10 个 Epoch 打印一次重构图（Original vs Reconstructed），让学生直观看到 AI 如何“学会”画旋涡。


* [ ] **模块 3：潜在空间操控 (炫酷交互)：**
* **Task A (插值)：** 实现两帧流体之间的潜在向量线性插值。
* **Task B (滑块控制)：** 使用 `ipywidgets.interact`。创建滑块对应前 4 个隐变量（Latent Variables），让用户拖动滑块，“捏”出不同的流场形态。


* [ ] **模块 4：异常检测挑战 (Science Task)：**
* 在测试集中人为加入噪声或遮挡。
* 展示 AE 如何通过“物理规律记忆”实现流场修复（Inpainting）。



## 📅 阶段四：收尾与文档说明 (Final Polish)

* [ ] **Markdown 注释：**
* 为所有关键参数（如 `stride`, `latent_dim`）增加中文注释。
* 增加“浙大 AI4S 公开课 - Shikai 老师”的水印说明。


* [ ] **Self-Test：** Agent 需自我运行一遍代码，确保无 `RuntimeError` 且 `ipywidgets` 渲染代码段正确。

---

## 💡 给 Agent 的特别指令 (System Prompt Add-on)

1. **美学要求：** 所有的流体图必须使用 `RdBu_r` 或 `viridis` 色图，背景设为深色或专业科研风格。
2. **逻辑要求：** 在 Notebook 末尾，对比 PCA（线性降维）和 AE（非线性降维）的优劣，升华 AI4S 的主题。
3. **输出要求：** 直接输出一个完整的 `.ipynb` 文件内容，或分段执行保存为文件。

---

Shikai 老师，你可以直接把这个文件投喂给 Agent。如果有任何具体代码层面的微调（比如你想加强贝叶斯张量分解的部分），只需告诉 Agent 在 `Phase 2` 中替换模型层即可！祝你的公开课大获成功！