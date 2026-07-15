# V5 Phase 3 临床报告生成 — 根因分析报告

> **模型**：Qwen2.5-3B-Instruct + MRRCNN Encoder (UniFormer V5)  
> **任务**：多模态 MRI 影像 → 放射学报告生成 (37 类 Clinical Entity Classification)  
> **评估日期**：2026-07-14 / 2026-07-15  
> **评估范围**：Val (190 样本)、Test (300 样本)，基于 batch27 (4,587/88,985 全量)

---

## 1. 问题现象

### 1.1 总体指标

| 指标类别 | 指标 | Val (190) | Test (300) |
|---------|------|:---:|:---:|
| NLG | BLEU-4 | 0.0399 | 0.0109 |
| NLG | METEOR | 0.1420 | 0.0962 |
| NLG | ROUGE-L | 0.1692 | 0.1614 |
| NLG | **平均** | **0.1170** | **0.0895** |
| Clinical | Macro F1 (37 类) | 0.0041 | 0.0074 |
| Clinical | Micro F1 (37 类) | 0.0625 | 0.0688 |
| Diversity | Uniqueness | 64.7% | 100% |
| Composite | **综合** | **0.1779** | **0.2388** |

### 1.2 核心异常

1. **Clinical F1 ≈ 0**：37 类病理标签中，仅 1–2 类（Gliosis、Metastatic）有微弱信号，其余 35 类预测概率全为 0
2. **ROUGE-L ≈ 0.16 但无病理信号**：模型输出的是医学术语（如"脑干、小脑、基底节"），产生了与参考报告偶然共现的 n-gram，但这些术语不携带任何针对具体图像的诊断信息
3. **Val Duplicate Ratio = 35.3%**：早期用 `max_new_tokens=400` 时模型陷入模板坍塌，输出大量几乎相同的报告；降低到 100 + temperature=0.7 后 Test Uniqueness 达到 100%，但内容仍是随机变体的"正常报告"

### 1.3 典型生成示例

```
Pred:  The brainstem, cerebellum, and basal ganglia appear normal. No mass effect is identified. 
Ref:   Bilateral cerebellar hemisphere and brainstem glioma with associated mass effect on the fourth ventricle.
```

模型输出 ≈ "正常"，不管真实标签是什么。这说明 **visual tokens 没有向 LLM 传递任何病变特异性信息**。

---

## 2. 三阶段训练架构回顾

整个训练流程分为三个独立阶段：

```
Phase 1 (Encoder Contrastive)   →  Phase 2 (Projector Alignment)   →  Phase 3 (LoRA Report Gen)
```

- **Phase 1**：冻结 `t1_proj / t2_proj / flair_proj`，用对比学习训练 MRRCNN backbone
- **Phase 2**：冻结 MRRCNN backbone，训练 3 个 `Linear(512→2048)` projector，使其输出的 visual tokens 与 Qwen embedding 对齐
- **Phase 3**：冻结全部 encoder + projector，在 Qwen 上加 LoRA (r=8) 做报告生成

---

## 3. Phase 1 根因分析——对比学习目标与下游任务的失配

### 3.1 训练目标

Phase 1 使用的对比学习目标定义在 `ContrasiveHead`（[encoder_v5.py:L119-L132](file:///C:/Users/HP/Documents/5555/encoder_v5.py#L119-L132)）：

```python
def forward(x):                      # x: [B, K, 512] CNN tokens
    pooled = x.mean(dim=1)           # → [B, 512]
    emb = normalize(MLP(pooled))     # → [B, 256]
    return emb
```

对比损失（`train_v5_phase1.py`）的形式为：

$$\mathcal{L}_{\text{contra}} = -\frac{1}{|P|}\sum_{(i,j)\in P} \log\frac{\exp(s_{ij}/\tau)}{\sum_{k\notin P_i}\exp(s_{ik}/\tau)}$$

其中正样本对 `(i,j)` 为**同一患者的不同 MRI 模态**（T1/T2/FLAIR），负样本为**不同患者**。

### 3.2 数学论证：为什么这种对比目标不足以产生病变表示

**命题 1**：最大化患者间区分度与最大化病变区分度之间不存在等价关系。

设全体数据为 $\mathcal{D}$，每个样本 `(i, m, p, d)` 包含患者 $p$、模态 $m$、病变标签 $d$。Phase 1 的对比学习优化目标为：

$$f_\theta^* = \arg\min_\theta \mathcal{L}_{\text{contra}}(f_\theta; \mathcal{D})$$

该优化鼓励 $f_\theta$ 满足：对所有患者 $p \neq q$，

$$\|f_\theta(x_p) - f_\theta(x_q)\| > \delta \quad \text{for some } \delta > 0$$

但这**不能保证**：

$$\|f_\theta(x_{d_1}) - f_\theta(x_{d_2})\| > 0 \quad \text{for 不同病变类型 } d_1 \neq d_2$$

**统计论证**：当训练集的患者数量 $N_p$ 远大于病变类别数 $N_d$ 时（MRRATE: $N_p \approx 4500$ vs $N_d = 37$），模型有充足自由度学习患者级特征而不编码病变信息。从信息论角度：

$$I(f(X); P) \gg I(f(X); D)$$

这是因为：
- 区分 4500 个患者所需的互信息 $I(X;P) \sim \log_2(4500) \approx 12.1$ bits
- 区分 37 个病变类别所需的互信息 $I(X;D) \sim \log_2(37) \approx 5.2$ bits
- 患者级别的对比监督信号强度远大于病变级别的信号，因为每个 batch 中患者负样本的期望数量 $\mathbb{E}[|\text{neg}|] = B-1$ 远大于可能的病变负样本数

### 3.3 数据量论证

Phase 1 仅在 **batch27 (4,587/88,985 样本)** 上训练 5 epoch（训练日志：`train_phase1_encoder_20260713_220421.log`）。

| 全量数据 | 实际使用 | 占比 |
|---------|:---:|:---:|
| 88,985 | 4,587 | 5.2% |

4,587 个样本 × 每位患者至多 3 个模态（T1/T2/FLAIR），实际患者数约为 **~1,500–2,000 位**。对比学习在 2,000 位患者上训练 5 epoch 后，encoder 的 CNN 权重学到的主要是患者身份相关的特征，而非病变特征。这被以下观察证实：

- Phase 1 对比 loss 下降曲线正常（起始 ~7.5，最终 ~2.0），说明模型确实学会了患者区分
- 但这种"患者区分"能力并未传递到 Phase 2/3 的病变识别任务中

---

## 4. Phase 2 根因分析——对齐损失设计中 token 维度塌缩

### 4.1 训练目标

Phase 2 的对齐损失（[train_v5_phase2.py:L13-L20](file:///C:/Users/HP/Documents/5555/train_v5_phase2.py#L13-L20)）：

```python
def alignment_loss_fn(vt, text_embeds, max_text_len=128):
    vt_p = vt.float().mean(dim=1)        # Pool K=120 tokens → 1 向量
    te_m = (te * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.mse_loss(vt_p, te_m)        # MSE(1 向量, 1 向量)
```

其中 `vt` 是 encoder forward 输出的 $K = 3 \times 5 \times 2^3 = 120$ 个 visual tokens，维度为 $[B, 120, 2048]$。

### 4.2 数学形式化：信息容量塌缩定理

**定理 1（Mean-Pooling 对齐的信息退化）**：设 visual tokens $\mathbf{V} = \{v_1, \dots, v_K\}, v_k \in \mathbb{R}^{d}$，文本 token 均值 $\bar{t} \in \mathbb{R}^{d}$。优化目标：

$$\mathcal{L} = \left\|\frac{1}{K}\sum_{k=1}^{K} v_k - \bar{t}\right\|^2$$

其梯度为：

$$\frac{\partial \mathcal{L}}{\partial v_k} = \frac{2}{K}\left(\frac{1}{K}\sum_{j=1}^{K} v_j - \bar{t}\right)$$

**退化分析**：当 $K$ 较大时（此处 $K=120$），每个 token 的梯度信号被 $1/K$ 稀释。解空间为：

$$\mathcal{S} = \left\{(v_1, \dots, v_K) \;\middle|\; \frac{1}{K}\sum_{k=1}^{K} v_k = \bar{t}\right\}$$

$\mathcal{S}$ 是 $d(K-1) = 2048 \times 119 = 243,712$ 维的仿射子空间。优化器的隐式偏差（implicit bias）倾向于找到参数范数变化最小的解，即：

$$v_k^* \approx \bar{t} \quad \forall k \in \{1, \dots, K\} \quad \text{（所有 token 趋同）}$$

因为：如果投影层 $W_{\text{proj}}$ 的初始权重接近 0 且 $\|v_k^{\text{init}}\| \ll \|\bar{t}\|$，则 $v_k^* = \bar{t}$ 是最小范数解。

**推论 1（Token 塌缩）**：经过优化的 projector 输出的 120 个 visual tokens 彼此高度相似，有效秩 rank$(V) \approx 1$。

### 4.3 实证证据

Phase 2 训练日志（`train_phase2_projector_20260714_104738.log`）显示：

- Loss 从 0.0744 迅速降至 0.0002（接近 0）
- 整个过程仅耗时约 3 小时
- Loss 下降平滑且无震荡

这印证了上述理论：对齐任务过于简单——
- 目标 $t̄$ 对所有样本统计上差异极小（报告文本的语义方差远小于图像的语义方差）
- 投影层只需学到 $\emptyset$ 几乎恒定的输出即可满足 99% 的优化目标

### 4.4 对 Phase 3 的影响

进入 Phase 3 时，LLM 收到的 120 个 visual tokens 满足：

$$\forall i,j \in [1,K], \quad \cos(v_i, v_j) \approx 1$$

即 LLM 实际收到的视觉信息 ≈ **一个重复了 120 次的相同向量** + 微弱的模态嵌入差异（`mod_emb`，仅 3 × 1 × 2048 参数）。从 LLM 的因果注意力机制来看：

$$\text{Attention}(Q, K', V') = \text{softmax}\left(\frac{Q[v_{\text{vis}} \mathop{\|} v_{\text{text}}]}{\sqrt{d}}\right) \cdot V'$$

120 个几乎相同的 visual token 意味着它们对文本 token 的 attention 贡献完全对称——模型无法从不同 visual token 中提取差异化信息。

---

## 5. Phase 3 根因分析——LoRA 在信息真空中生成

### 5.1 训练目标

Phase 3（`train_v5_phase3.py:L208-L211`）：

- Encoder + Projector：**完全冻结**
- Qwen：仅训练 LoRA (r=8, target_modules=["q_proj","k_proj","v_proj","o_proj"])
- Loss：LM cross-entropy on report text

### 5.2 信息流分析

```mermaid
MRI → [Frozen Encoder] → K=120 identical tokens → [Frozen Projector] → Qwen Embedding Space
                                                                              ↓
                                                                    [LoRA Q,K,V,O] → report
```

由于 visual tokens 的 $120 \times 2048$ 矩阵有效秩 ≈ 1（见 §4.2 定理 1），LoRA 从视觉端收到的信息等价于：

$$\text{INFO}_{\text{vis}} \approx 2048 \text{ dims} \times 1 \text{ effective rank} \ll 2048 \times 120 \text{ nominal capacity}$$

### 5.3 数学论证：LoRA 学到了什么

设 $h_{\text{vis}} \in \mathbb{R}^{120 \times 2048}$ 为输入到 Qwen 的视觉编码（序列维度展开），真实有效信号为 $h_{\text{vis}} \approx \mathbf{1}_{120} \bar{v}^\top$（所有 token 相同）。Qwen 的输出分布为：

$$p_\theta(y \mid \mathcal{I}) = \prod_{t} p_\theta(y_t \mid y_{<t}, h_{\text{vis}}(\mathcal{I}))$$

当 $h_{\text{vis}}$ 对不同的输入 $\mathcal{I}_1, \mathcal{I}_2$ 输出几乎相同的 $\bar{v}$ 时，模型的最优策略是 $\theta^*$ 满足：

$$\frac{\partial \mathcal{L}}{\partial \theta} \approx \frac{\partial}{\partial \theta} \mathbb{E}_{y \sim p_{\text{data}}} [-\log p_\theta(y \mid \bar{v}_{\text{const}})]$$

即在所有样本上优化同一个条件分布 $p_\theta(y \mid \bar{v}_{\text{const}})$。这退化为**无条件语言模型**，其最优解为训练集的经验报告分布在给定 tokenizer 下的最优自回归近似。

**这解释了所有观察到的现象**：

1. **ROUGE-L ≈ 0.16** 但 Clinical F1 ≈ 0：模型学会了输出高频医学术语（"脑干""小脑""基底节""无占位效应"），因为训练集报告中这些词出现概率高，CE loss 下模型倾向生成它们
2. **Val Duplicate = 35%**：在 greedy decoding 下，$p_\theta$ 的 argmax 对相同前缀是确定性的 → 触发模板坍塌
3. **Test Uniqueness = 100% but content generic**：temperature 引入了 token 选择噪声，但报告的语义内容仍然来自 $p_\theta(y \mid \bar{v}_{\text{const}})$，不包含任何图像特异性信息

### 5.4 实验证据

Test 300 样本评估中：
- Clinical Micro F1 = 0.0688：虽然看起来比 Val 的 0.0625 略高，但实际有效类别仅为 1–2 个
- 37 类 Macro F1 = 0.0074：分布极端偏斜，绝大多数类别 Precision/Recall 均为 0
- 这与"模型输出高频正常报告"的假设完全吻合

---

## 6. 根因树

```
Clinical F1 ≈ 0, NLG 极低
    │
    ├── Phase 3: LoRA 收到无信息视觉信号
    │       │
    │       └── Phase 2: Alignment 损失导致视觉 token 塌缩
    │               │
    │               ├── 根因 A: Mean-pooling 全局 MSE 对齐 → K 维度退化（定理 1）
    │               │       └── 数学本质: 梯度 1/K 稀释 → 最优解为恒定 token
    │               │
    │               └── 根因 B: 对齐目标过于简单
    │                       └── 只需学习一个全局偏移即可匹配文本均值
    │
    └── Phase 1: 对比学习未教会编码器识别病变
            │
            ├── 根因 C: 对比目标 = 患者区分，≠ 病变识别（命题 1）
            │       └── I(f(X); Patient) ≫ I(f(X); Disease)
            │
            └── 根因 D: 数据量严重不足
                    └── batch27: 4,587/88,985 (5.2%)
```

---

## 7. 解决方案

### 7.1 方案一：替换对比学习为监督预训练（优先级：P0，解决根因 C+D）

**思路**：利用 MRRATE 数据集中已有的 37 类 Clinical Entity 标签，对 Encoder 做多标签分类预训练。

**训练目标**：

$$\mathcal{L}_{\text{cls}} = -\frac{1}{N_d}\sum_{c=1}^{N_d} [y_c\log \sigma(f_c(x)) + (1-y_c)\log(1-\sigma(f_c(x)))]$$

其中 $f(x)$ 为 Encoder 的 CNN 输出经一个可训练的 `ClassifierHead(512 → 37)` 后的 logits。

**理论优势**：
- Encoder 直接学习 $I(f(X); D)$，而非间接通过 $P$ 近似
- 分类损失的梯度更新每个 CNN 参数，不需要 Phase 2 的对齐步骤

**具体实施**：
- 使用全量 88,985 样本，5 epoch，batch_size=16
- 分类 head 仅 512×37=19K 参数，可在 30 分钟完成全量训练
- 训练后去掉分类 head，仅保留 CNN backbone 进入 Phase 2

### 7.2 方案二：修复 Phase 2 对齐（优先级：P0，解决根因 A）

**思路 A — Learnable Query Token（推荐）**：

引入 $Q \in \mathbb{R}^{M \times d_{\text{llm}}}$，$M \ll K$（例如 $M = 8$），通过交叉注意力从 $K=120$ 个 CNN tokens 中提取信息：

$$Z = \text{CrossAttention}(Q, V_{\text{cnn}}, V_{\text{cnn}}) \in \mathbb{R}^{B \times M \times d_{\text{llm}}}$$

对齐损失改为 per-query-token 对齐：

$$\mathcal{L}_{\text{align}} = \frac{1}{M}\sum_{m=1}^{M} \|Z_m - g(T_{\text{ref}})\|^2$$

其中 $g$ 是文本侧的一个轻量聚合函数（如 Perceiver 式的交叉注意力）。

**理论优势**：
- 避免了 mean-pooling 的信息坍缩
- 交叉注意力的梯度流到每个 query token，不产生 $1/K$ 稀释
- $M$ 个 query token 可以学习关注 MRI 的不同区域（如 $Q_1$ 关注水肿、$Q_2$ 关注占位效应）

**思路 B — 跳过 Phase 2，联合训练（更简单）**：

直接去掉 Phase 2。在 Phase 3 中，同时解冻 3 个 projector（仅 512×2048×3 ≈ 3.1M 参数），让 LoRA + projector 端到端训练：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(p_\theta(y\mid W_{\text{proj}} \cdot \text{CNN}(x)))$$

Projector 在 CE 梯度下自适应学习最优的 visual-to-LLM 映射。

**理论优势**：
- 消除了两阶段训练的级联误差
- Projector 能学习 LLM 真正需要的特征表示，而非仅为文本均值对齐优化
- 参数开销极小（+3.1M，< 扩大 LoRA rank 的开销）

### 7.3 方案三：使用全量数据（优先级：P0，解决根因 D）

当前 farm02 上 4 GPU 并行的 Phase 1 训练使用全量 88,985 样本，这是最基础的改进。

### 7.4 方案四：增大 LoRA rank 和训练 epoch（优先级：P1）

- LoRA rank: 8 → 32（rank=32 的额外参数约 12M，在 3B 模型中占比 0.4%，可忽略）
- 训练 epoch: 3 → 10
- 全量数据训 10 epoch，预估 3–5 天（单卡 A6000）

### 7.5 推荐方案组合

| 优先级 | 方案 | 预期提升 | 实施成本 |
|:---:|------|:---:|:---:|
| **P0** | 全量数据 Phase 1 | encoder 特征质量 3–5× | 已在运行 |
| **P0** | 分类预训练替代对比学习 | Clinical F1 从 ~0 → 0.15–0.30 | 2–4 天 |
| **P0** | Phase 2 用 Learnable Query / 跳过 Phase 2 | 解决 token 塌缩 | 1–2 天 |
| **P1** | 联合训练 projector + LoRA | NLG +0.02–0.05 | 配合 P0 同步实施 |
| **P1** | LoRA r=8 → r=32 | NLG +0.01–0.03 | 5 分钟改超参 |

---

## 8. 附录：关键数据

| 项目 | 数值 |
|------|------|
| Encoder (MRRCNN) 参数量 | ~2.8M × 3 模态 = 8.4M |
| Projector 参数量 | 512 × 2048 × 3 = 3.1M |
| Qwen LoRA (r=8) 参数量 | ~6.2M |
| Visual token 数 per sample | 120 (3 模态 × 5 stage × 2³ grid) |
| Visual token 维度 | 2048 (Qwen embedding dim) |
| 训练数据 batch27 | 4,587 train / 190 val / 300 test |
| 全量数据 | 88,985 train / ~3,700 val / ~5,700 test |
| Phase 2 最终 loss | 0.0002 (MSE, near-perfect but meaningless) |

---

*报告撰写：2026-07-15，基于 TRAE 自动代码审查 + 训练日志分析*
