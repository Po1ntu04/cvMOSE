# SAM2 / SAM3 论文深读与 MOSEv2 作业优化路线

> 目的：把 SAM2、SAM3/SAM3.1 论文理解、已有 SAM3 分析报告、15 个待测视频真实观察、当前预测行为统一成可回忆的持久化认识，并据此判断后续优化方向。

## 0. 证据来源

本记录基于以下本地文件，不依赖记忆臆测：

- `homework/SAM2.pdf` → 已抽取为 `homework/analysis/paper_text/SAM2.txt`。
- `homework/SAM3.pdf` → 已抽取为 `homework/analysis/paper_text/SAM3.txt`。
- `homework/analysis/SAM3-analyze.md`：已有 SAM3 深度分析报告，本轮用论文正文复核其核心判断。
- `MOSEv2 作业说明.pdf` → 已抽取为 `homework/analysis/paper_text/MOSEv2_assignment.txt`。
- `homework/analysis/mosev2_15_video_observations.md`：15 个待测视频语义观察与联系图记录。
- `homework/analysis/mosev2_15_video_pred_area_summary.json`：SAM2、SAM3.1 adapter、SAM3.1 public-simple 的空帧/面积统计。
- `homework/logs/sam31_run_report.md`、`homework/logs/sam31_public_simple_run_report.md`：SAM3.1 两条路线的运行结果。

关键证据索引：

- 作业要求是“完整视频序列 + 首帧 mask → 推理完整视频”，可用 SAM2/SAM3/其他 VOS：`paper_text/MOSEv2_assignment.txt:13-19`。
- 评分以 `J&F'` 为主，SAM2.1-B+ 跑完 15 个视频预估从 41.05 提到 43.25，超过 44 有 bonus：`paper_text/MOSEv2_assignment.txt:68-72`。
- SAM2 定义 PVS：点/框/mask 提示任意帧，输出目标的时空 masklet：`paper_text/SAM2.txt:46-49`, `154-188`。
- SAM2 架构核心是 streaming memory、memory attention、memory bank、object pointer 和 occlusion head：`paper_text/SAM2.txt:195-206`, `233-264`, `880-920`。
- SAM2 论文明确承认长遮挡、拥挤相似物、快速细小物、无对象间通信是难点：`paper_text/SAM2.txt:843-854`。
- SAM3 定义 PCS：简单 noun phrase / image exemplar / 二者结合，检测、分割、跟踪所有概念实例：`paper_text/SAM3.txt:58-67`, `103-122`。
- SAM3 架构核心是 detector + SAM2-style tracker，并用 presence token 解耦 recognition / localization：`paper_text/SAM3.txt:68-74`, `156-164`。
- SAM3 在视频中用 detector 修正 tracker：masklet detection score、confirmation delay、periodic/detection-guided re-prompting：`paper_text/SAM3.txt:202-213`, `1156-1237`。
- SAM3 在 VOS 上论文报告强于 SAM2，MOSEv2 val 从 SAM2.1 L 的 47.9 到 SAM3 的 60.3：`paper_text/SAM3.txt:438-454`。
- SAM3 局限：简单名词短语、短视频、fine-grained OOD 概念、对象数线性成本、缺少共享 object-level context：`paper_text/SAM3.txt:980-1002`。
- SAM3.1/Object Multiplex：把多个对象放入 fixed-capacity buckets 共享 memory 路径，MOSEv2 从 60.3 到 62.3，但公开 benchmark 有 mixed results：`paper_text/SAM3.txt:3183-3282`。

---

## 1. 本作业问题的形式化

给定 15 个视频：

\[
V = (I_0, I_1, \dots, I_{T-1}),
\]

以及首帧实例标注：

\[
Y_0 \in \{0,1,\dots,K\}^{H\times W},
\]

其中每个非零 id 表示一个**特定实例**，而不是类别。目标是输出：

\[
\hat{Y}_{0:T-1},
\]

要求每个 id 在全视频中保持同一实例身份。该问题属于 **semi-supervised VOS / PVS 的首帧 mask 特例**，而不是开放词汇概念分割。它的关键约束是：

1. **没有文本概念输入**；
2. **首帧 mask 是强实例锚点**；
3. **如果目标不可见，应该倾向输出空/不可见，而不是切到同类目标**；
4. **目标重现后需要恢复同一实例，而不是找到同类类别中最显眼实例**；
5. **最终只看提交 mask 的 J&F'，不奖励模型路线本身是否“更先进”**。

这一定义决定了：SAM2 的原生 PVS/VOS 链路与作业最匹配；SAM3 的 PCS 能力只有在能有效提供概念/正负 exemplar/检测重锚定时才有额外价值。

---

## 2. SAM2 论文阅读

### 2.1 Task：SAM2 旨在解决什么问题？

SAM2 要解决的是 **Promptable Visual Segmentation (PVS)**：给定图像或视频，以及一个或多个视觉提示（点、框、mask），输出被提示目标在空间或时空中的分割结果。视频里输出的是 masklet，即同一目标跨帧的 mask 序列。

形式化：给定视频

\[
V=(I_t)_{t=0}^{T-1},
\]

提示集合

\[
P=\{(t_j, p_j)\}_{j=1}^{m},
\]

其中 \(p_j\) 可以是点、框、mask，目标是估计

\[
M=(m_t)_{t=0}^{T-1},\quad m_t\in\{0,1\}^{H\times W},
\]

并允许用户在任意帧继续追加提示修正。传统 semi-supervised VOS 是 PVS 的特例：只在首帧给一个 GT mask。

本作业正是这个特例。

### 2.2 Challenge：过往方法遇到什么挑战？

SAM2 论文指出，视频比图像多了时间维度，目标会出现：

- 外观变化；
- 形变；
- 遮挡；
- 光照变化；
- 摄像头运动、模糊、低分辨率；
- 长视频高效处理压力。

对传统“SAM + 外部 tracker”路线来说，问题在于：SAM 能在单帧分割，但 tracker 不一定适配任意对象；tracker 失败后往往只能重标当前帧并重启，不能自然利用历史记忆。SAM2 的 PVS 目标就是把“单帧交互分割”和“跨帧传播/修正”做成统一模型。

### 2.3 Insight & Novelty

#### 2.3.1 Inspiration

SAM2 的灵感来自三条线：

1. **SAM 的 promptable segmentation**：点/框/mask 可以定义用户意图；
2. **VOS 的首帧 mask 条件传播**：首帧监督可以变成目标 identity 锚点；
3. **交互式标注数据引擎**：模型失败处由人修正，再反馈训练。

#### 2.3.2 Insight

**Insight A：视频分割不是每帧重新分割，而是“当前帧视觉特征 + 目标历史记忆”的条件预测。**  
对应结构：image encoder 提供每帧 unconditioned embedding；memory attention 让当前帧 cross-attend 到 memory bank。

**Insight B：首帧或被提示帧应该长期保留，近期未提示帧只保留有限窗口。**  
对应结构：memory bank 包含 prompted frames 和最近 N 帧 FIFO memory；这解释了为什么首帧 GT mask 对作业非常关键。

**Insight C：不可见本身是合法状态。**  
对应结构：occlusion / object-present head；如果目标被遮挡或出画，模型可以输出无目标，而不是强制分割某个相似物。

**Insight D：困难同类目标需要让模型学习“不是只看外观”。**  
对应训练：mosaic 2×2 augmentation 把相同/相似视频拼在一起，让目标变小且出现相似干扰，迫使模型用运动/时序连续性区分目标。

#### 2.3.3 Novelty

SAM2 的 novelty 集中在：

1. **统一图像和视频的 promptable model**：图像可视为单帧视频；
2. **streaming memory 架构**：逐帧处理，适配长视频；
3. **memory attention + memory encoder + memory bank**：把目标历史状态作为当前帧条件；
4. **object pointer token**：用轻量 token 表示高层目标身份；
5. **occlusion head**：允许目标不可见；
6. **SA-V 数据引擎和大规模视频 masklet 数据**；
7. **交互式 refinement 作为一等任务**：追加提示不需要重启整个 tracker。

### 2.4 Potential flaw

#### 2.4.1 情境局限与可扩展方向

SAM2 假设用户通过视觉提示已经定义了“哪个对象”。它不负责开放词汇概念发现，也不理解“草莓后面那块被手拿走的切片”这类语言关系。它可以通过额外提示恢复，但在本作业提交时我们不能向模型提供真实后续 GT。

可扩展方向：

- 自动选择可信中间帧作为 pseudo prompt；
- 引入显式运动模型；
- 多目标共享上下文与竞争抑制；
- 结合高层语义/MLLM 做遮挡后重锚定。

#### 2.4.2 哪些数据性质会特别困难？

SAM2 论文自己的限制与我们 15 个视频高度重合：

- long occlusion；
- crowded similar objects；
- fast-moving thin/fine objects；
- extended videos；
- multi-object independent inference 缺少对象间通信。

具体到本作业：`r13u5z4y`、`q0sizv6m`、`msinig6m`、`amfdu83t`、`lcgc29va` 等正是这几类困难的组合。

#### 2.4.3 哪个困难值得写成 paper？

最值得深挖的是：**single-initial-mask VOS 中的遮挡后实例重锚定**。

原因：它不只是“mask 边界不准”，而是“首帧只给了一个残缺可见区域，目标完全消失后，如何在多个相似实例中重新确认同一对象”。这同时涉及 memory selection、occlusion reasoning、identity re-identification、hard negative 同类抑制，是 MOSEv2 的核心痛点。

---

## 3. SAM3 / SAM3.1 论文阅读

### 3.1 Task：SAM3 旨在解决什么问题？

SAM3 提出 **Promptable Concept Segmentation (PCS)**：给定图像或短视频，以及概念提示，输出所有匹配该概念的实例 mask；若输入是视频，还要输出跨帧一致的 identity。

概念提示可以是：

- 简单 noun phrase，例如 “yellow school bus”；
- image exemplar，正/负框；
- 文本 + exemplar 的组合。

形式化：

\[
X\in\{\text{image},\text{short video}\},\quad P=(q,E^+,E^-),
\]

输出：

\[
\mathcal{O}=\{(m_{k,t}, id_k)\}_{k=1}^{N}.
\]

关键是：SAM3 要找出**所有属于概念的实例**，而不是只追踪首帧指定的一个实例。

### 3.2 Challenge：过往方法遇到什么挑战？

**挑战 A：recognition 与 localization 冲突。**  
判断“图里有没有这个概念”需要全局上下文；定位“这个 query 在哪里”需要局部几何精度。让每个 object query 同时承担二者，会造成误检/漏检。

**挑战 B：detector 与 tracker 冲突。**  
detector 应该 identity-agnostic：所有“猫”都应被找出；tracker 必须 identity-sensitive：两只猫不能混。一个表示同时承担两种目标会冲突。

**挑战 C：开放概念天然歧义。**  
简单 noun phrase 也可能多义、主观、边界模糊、被遮挡或模糊。单一 GT/单一输出会惩罚合理但不同的解释。

**挑战 D：数据不可得。**  
PCS 需要“图像/视频 + noun phrase + 所有实例 mask + hard negatives + exhaustivity verification”，普通数据集不满足。

**挑战 E：视频中的遮挡、同类干扰、检测/跟踪互相纠错。**  
纯 tracker 会漂移，纯 tracking-by-detection 又容易关联不稳；必须同时处理新实例发现、旧实例延续、重复 masklet、假阳性和 occlusion。

### 3.3 Insight & Novelty

#### 3.3.1 Inspiration

SAM3 的灵感来自：

1. SAM/SAM2 的 promptable segmentation 能力；
2. DETR / MDETR / open-vocabulary detection 的概念条件定位路线；
3. SAM2 的 memory-based video tracker；
4. 数据引擎与 hard-negative/exhaustive annotation 的经验；
5. 视频 VOS 中遮挡后漂移、同类干扰和多目标关联失败。

#### 3.3.2 Insight

**Insight A：PCS 不是纯分割，而是“概念存在性 + 实例定位 + 像素 mask”的分层问题。**  
对应结构：presence token / presence head，将

\[
p(q_i \text{ matches NP})
\]

拆成

\[
p(q_i \text{ matches NP}\mid NP \text{ present})\cdot p(NP \text{ present}).
\]

**Insight B：视频 PCS 应该解耦 detector 与 tracker。**  
detector 负责“概念实例发现”，tracker 负责“已发现实例的 identity 延续”。这就是 SAM3 的 detector + SAM2-style tracker。

**Insight C：concept-level interactivity 和 instance-level interactivity 是互补的。**  
正/负 exemplar 可以修正整个概念类的误检/漏检；点击/点框则修单个 masklet 的边界。

**Insight D：开放词汇需要 hard negatives 和 exhaustivity verification。**  
没有大量负样本，模型会“看着像就报一个”；没有 exhaustivity，模型无法学会找全所有实例。

**Insight E：视频跟踪必须允许 detector 反向纠正 tracker。**  
对应结构：masklet detection score、track confirmation delay、duplicate removal、periodic re-prompting、detection-guided re-prompting。

**Insight F：多对象不应完全独立处理。**  
SAM3.1/Object Multiplex 把多个对象装入固定容量 bucket，共享 memory encoding / retrieval，用 object-specific embedding 保持身份。这是对 SAM2/SAM3 原始“每对象独立 memory”局限的直接回应。

#### 3.3.3 Novelty

SAM3 的 novelty 是完整闭环：

1. **PCS 任务定义**：概念提示下发现、分割、跟踪所有实例；
2. **DETR-style detector + SAM2-style tracker 的双系统架构**；
3. **presence head**：显式解耦全局概念存在性与局部定位；
4. **正/负 image exemplar prompting**：让交互从单实例修补扩展到概念层纠错；
5. **ambiguity head**：用专家分支处理多义概念；
6. **temporal disambiguation**：用 detector 纠正 tracker；
7. **SA-Co 数据引擎和 benchmark**：大规模 noun phrase、hard negatives、exhaustivity；
8. **SAM3.1/Object Multiplex**：共享多对象 memory 路径，提高多目标效率，并在 MOSEv2 上提升。

### 3.4 Potential flaw

#### 3.4.1 情境局限与可扩展方向

SAM3 的主任务是 PCS，而不是只给首帧 mask 的实例级 VOS。它被限制在简单 noun phrase 和短视频；复杂关系语言要外接 MLLM；fine-grained OOD 概念 zero-shot 较弱；原始视频成本随对象数线性增长，3.1 用 multiplex 缓解。

对本作业而言，最大局限是：**PCS 的“找出所有同概念实例”与作业的“只追踪首帧指定实例”存在目标不一致**。例如 “strawberry slice” 会找到多个草莓切片，但作业只要被首帧 mask 指定的那一块。

#### 3.4.2 哪些数据性质会特别困难？

对 SAM3/SAM3.1，最难的是：

- 首帧 mask 是残缺局部，但没有 noun phrase；
- 多个同类实例几乎相同，概念检测会找出所有实例，反而弱化单实例身份；
- 目标很小，detector/presence 可能认为不存在；
- 目标在边缘或短暂出画，confirmation/suppression 可能过度保守；
- 不提供正负 exemplar 时，模型没有办法知道“不是那些同类干扰物”。

#### 3.4.3 哪个困难值得写成 paper？

如果从 SAM3 方向写 paper，最值得的是：**从单首帧 mask 自动构造 concept + positive/negative exemplars，并用于遮挡后实例重锚定**。

这不是简单“把 SAM3 跑在 VOS 上”，而是把 SAM3 的 concept detector 变成一个 re-identification oracle：根据首帧目标和场景中的同类干扰，自动生成目标概念、负例同类、空间/时间一致性约束，再把 detector 的候选用于纠正 tracker。

---

## 4. 两篇论文对本作业的关键启发

### 4.1 本作业更接近 SAM2 的核心定义

作业给的是首帧 mask，不是 noun phrase。目标是同一实例的 masklet。这与 SAM2 的 PVS/VOS 定义完全对齐。

因此，**SAM2 不是“旧模型所以差”，而是任务接口最匹配**：

- 输入：首帧 mask；
- 输出：同一对象全视频 masklet；
- 机制：首帧 prompted memory + 近期 memory + occlusion head；
- 失败点：长遮挡/相似物/小目标，正是我们要局部优化的点。

### 4.2 SAM3 理论上更强，但当前调用方式没有用到它的强项

SAM3/SAM3.1 论文里强的部分包括 concept detection、presence head、正负 exemplar、detector-guided re-prompting、Object Multiplex。但我们当前 adapter 是把首帧 GT mask 注入内部 tracker，基本上绕过了 text/exemplar detector 的概念发现与 re-prompt 能力。

因此它变成了“更重的 SAM2-style tracker”，却不一定比 SAM2 更适合小目标首帧 mask VOS。

当前运行证据也支持这一点：

- SAM3.1 GT-mask adapter：1004 帧，约 5.027 fps，峰值约 8.77 GiB allocated / 9.48 GiB reserved；
- SAM2 旧 memory probe：约 966.5 MiB allocated / 1144 MiB reserved；
- SAM3.1 public-simple：形式上能提交，但 1004 帧中只有 4 帧有非空内部输出，应视为失败路线；
- 预测统计中，SAM3.1 adapter 在多数视频空帧显著多于 SAM2，例如 `r13u5z4y` 39/45 空、`q0sizv6m` 30/42 空、`8jsm23a7` 31/49 空。

### 4.3 SAM3 论文中的“好思想”仍可借给 SAM2 路线

即使主模型选择 SAM2，SAM3 的几个思想仍然有价值：

1. **presence / occlusion 分离**：先判断目标是否应存在，再决定 mask；
2. **detector-guided re-prompting**：用可靠候选修正 tracker memory；
3. **confirmation delay**：不要立刻相信短暂出现的新候选；
4. **duplicate / distractor suppression**：同类目标密集时，宁愿延迟确认，也不要切错；
5. **positive/negative exemplar 思想**：对同类干扰要显式压制，而不是只增强目标。

这些可以工程化为 SAM2 上的自监督 re-anchor / postprocess，而不必把整条路线切到 SAM3。

---

## 5. 结合 15 个视频的系统性问题图谱

### 5.1 真实痛点排序

按对最终分数的潜在影响和可修复性排序：

1. **遮挡后 identity switch**：`r13u5z4y`、`q0sizv6m`、`msinig6m`。
2. **小目标被下采样/阈值吞掉**：`lcgc29va`、`amfdu83t`、`1qlssuz2`、`4vznweiu`、`8jsm23a7`。
3. **边缘/出画导致不可见状态难判断**：`amfdu83t`、`c8lutf29`、`pe0d85lk`、`q0sizv6m`、`2smf7uq9`。
4. **多同类/多实例竞争**：车辆、鸟、狮子、考拉、豚鼠、草莓、字母方块、游戏牌。
5. **重复纹理和低对比背景**：道路、红黑座椅/Logo、水族箱、水下砂纹、白底白块。
6. **多目标 id 合并/互换**：`4f98052b`、`msinig6m`、`pe0d85lk`、`q0sizv6m`。

### 5.2 当前预测行为提示

从 contact sheet 和空帧统计看：

- SAM2 通常比 SAM3.1 adapter 更愿意持续输出，但持续输出不保证 identity 正确；
- SAM3.1 adapter 更保守，很多视频快速变成空 mask；
- SAM3.1 public-simple 已基本可以舍弃；
- 某些视频 SAM3.1 adapter 有潜在参考价值，例如 `3epdtmyr` 全程非空，`z6dx46qr` 比 SAM2 少一些空帧，但仍需语义核验，不能盲目替换。

---

## 6. 优化改进路线：倾向从 SAM2 出发

### 6.1 总判断

**主路线应从 SAM2 出发优化，而不是从 SAM3 public 或 SAM3.1 adapter 出发重写。**

原因：

1. 作业输入形式与 SAM2 原生 PVS/VOS 完全对齐；
2. SAM3 的核心强项 PCS 在本作业没有天然 prompt；
3. public 方式实测几乎全空，明确舍弃；
4. adapter 方式虽然形式上跑通，但更重、更保守，且用户实测全量分数略低于 SAM2；
5. 当前真正可提升的不是“换模型”，而是对 15 个困难视频做 occlusion-aware、identity-aware、small-object-aware 的局部修复。

### 6.2 第一优先级：保留 SAM2 基线，做逐视频选择性修复

不要整体替换 SAM2 输出。应以 SAM2 为默认提交，对每个视频/帧段判断是否替换或修正。

建议建立表：

| 视频 | 默认 | 可能修复点 | 是否可用 SAM3 adapter 辅助 |
|---|---|---|---|
| `1qlssuz2` | SAM2 | 小车末尾空/漂移，尝试 crop/小目标增强 | 仅作参考，SAM3 后段空多 |
| `2smf7uq9` | SAM2 | 鸟类同类干扰和边缘，核验恢复帧是否同一只 | SAM3 空多，不优先 |
| `3epdtmyr` | 对比选择 | SAM2 空多，SAM3 非空但需核验是否同一狮子 | 可重点比较 |
| `4f98052b` | SAM2 | 长段空、摄像机转向、多目标；可能只能有限修 | SAM3 空更多，不优先 |
| `4vznweiu` | SAM2 | 小白块，SAM2 已相对稳定 | SAM3 可参考但必要性低 |
| `8jsm23a7` | SAM2 | 手遮挡后的同牌身份 | SAM3 空多，不优先 |
| `amfdu83t` | 谨慎 | 边缘袋鼠极小，可能多数不可见；需要确认是否应空 | SAM3/SAM2 都弱 |
| `c8lutf29` | 谨慎 | 红衣人边缘/快速运动，避免切到其他红色目标 | SAM3 基本空 |
| `jadgtmfl` | SAM2/对比 | 小鱼与珊瑚遮挡，SAM2/SAM3 都可参考 | 可比较末尾空 |
| `lcgc29va` | SAM2 | 极小行人，考虑 crop/小目标增强 | SAM3 空多 |
| `msinig6m` | SAM2 + id约束 | 多考拉 id 互换/遮挡 | SAM3 空多 |
| `pe0d85lk` | SAM2 | 大目标但出画/尺度变化，修边界和出画 | SAM3 部分参考 |
| `q0sizv6m` | SAM2 + 强核验 | SAM2 非空但可能切豚鼠，需人工核验/约束 | SAM3 空多 |
| `r13u5z4y` | SAM2 + 专项修 | 草莓遮挡后重识别失败，是重点 | SAM3 基本丢失 |
| `z6dx46qr` | 对比选择 | 水下小目标，SAM2 空多，SAM3 中段可能有参考 | 可重点比较 |

### 6.3 第二优先级：SAM2 自举多锚点 re-prompt

思路：从 SAM2 预测中选出可信中间帧作为 pseudo mask prompt，重新跑 SAM2，使 memory 不只依赖首帧和近期错误预测。

候选帧选择规则：

- mask 面积与首帧/邻帧相比不过度突变；
- bbox 移动连续；
- 不与同类干扰明显重叠；
- 目标在原帧中语义可见；
- 对多目标视频，不与其他 id 冲突。

重跑策略：

1. 首帧 GT mask 固定；
2. 选择 1-3 个可信 pseudo prompts；
3. 从这些提示帧向前/向后传播；
4. 与原 SAM2 输出融合；
5. 只接受语义和几何一致的改进。

风险：如果 pseudo prompt 已经 identity switch，会污染 memory。因此 `r13u5z4y`、`q0sizv6m`、`msinig6m` 必须人工/语义核验后才能加入 pseudo prompt。

### 6.4 第三优先级：小目标 crop / 高分辨局部重跑

对 `lcgc29va`、`amfdu83t`、`1qlssuz2`、`4vznweiu`、`8jsm23a7`、`z6dx46qr`，全图输入下目标太小。可以尝试：

1. 根据首帧 bbox 和前几帧预测建立搜索区域；
2. 对局部 crop 放大后跑 SAM2；
3. 将局部 mask 映射回全图；
4. 与全图 SAM2 结果比较，仅在小目标明显恢复时替换。

这条路线的理论依据：SAM2 论文中输入分辨率提升对 MOSE/dev 和图像任务有明显收益；而本作业小目标面积常低于 1%，局部 crop 等价于提高目标相对分辨率。

### 6.5 第四优先级：occlusion-aware 输出门控

目标不可见时，输出错误同类目标通常比输出空更伤 identity。建议对每帧做门控：

- 面积突然扩大/缩小到不合理范围；
- bbox 突然跳到同类干扰处；
- 目标位于遮挡物内部或明显被手/人/边缘覆盖；
- 与另一个 id 或同类候选发生冲突；
- SAM2 与 SAM3 都低可信/空，且原帧语义不可见。

满足这些条件时，宁愿保留空 mask 或短暂延续上一可信 mask 的保守版本，而不是切换到显眼干扰物。

### 6.6 第五优先级：多目标 identity 约束

对 `msinig6m`、`q0sizv6m`、`pe0d85lk`、`4f98052b`：

- 同一帧不同 id mask 应互斥或按置信度裁剪 overlap；
- id 的空间轨迹不应交叉跳变，除非原帧语义确认；
- 每个 id 的面积和 bbox 应有连续性先验；
- 若两个 id 同时追到同一实例，保留更符合历史轨迹的 id，另一个置空或回退。

这是把 SAM3.1 Object Multiplex 的“对象间上下文”思想，用后处理形式移植到 SAM2 输出上。

### 6.7 SAM3 仍可作为辅助，但不作为主路线

可保留 SAM3.1 adapter 的用途：

1. 作为候选 mask 来源，与 SAM2 做帧段级比较；
2. 在 `3epdtmyr`、`z6dx46qr` 这类 SAM3 非空更连续的视频中，做语义核验后局部替换；
3. 用 raw score/空帧作为 occlusion 参考信号；
4. 未来若要研究，可尝试手动/自动 noun phrase + positive/negative exemplar 让 SAM3 detector 参与重锚定。

但当前不建议继续投入 public-simple 路线；它已经是有效的失败对照。

---

## 7. 最小可执行实验顺序

1. **生成每个视频的 frame-level audit 表**：记录 SAM2/SAM3 面积、bbox、空帧、突变点。
2. **优先核验 5 个最可能提分视频**：`r13u5z4y`、`q0sizv6m`、`3epdtmyr`、`z6dx46qr`、`lcgc29va`。
3. **先做不改模型的选择性融合/门控**：成本最低，最适合 15 视频作业。
4. **再做 SAM2 multi-anchor re-prompt**：只在语义核验后的可信帧使用 pseudo mask。
5. **最后尝试 crop 重跑**：用于极小目标，避免大规模改动。
6. **每次只生成一个新的提交 zip 并记录分数**：Codabench 每天 20 次，必须有实验日志。

---

## 8. 当前结论

从论文任务定义、架构细节、作业输入形式和当前实测输出四方面综合判断：

- **public 方式调用 SAM3.1 基本舍弃**；
- **SAM3.1 GT-mask adapter 可作为辅助候选，但不是主优化路线**；
- **主路线应以 SAM2 输出为基线，加入面向 15 个困难视频的遮挡门控、实例身份核验、小目标 crop、多锚点 re-prompt 和多目标互斥约束**；
- **真正的优化问题不是“哪个 foundation model 更先进”，而是“在只有首帧实例 mask 的限制下，如何防止长遮挡后切到同类目标，并在小目标/边缘目标上保持可见性判断正确”。**
