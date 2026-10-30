# 第十章 神经网络与深度学习：可学习表示的组合

> 往年资料：S16（概述），S17 4–47 页（多层感知机与 BP）、49–71 页（CNN），S18 3–27 页（RNN），S19 3–73 页（深度学习与前沿概览）。原始章节编号出现“第十一章／14.x”残留，本书按课程大纲统一为第十章。历史产品列表与性能数据未当作最新事实。

> **本章定位（编者说明）**：本章属于“往年资料”，公式与机制说明依据 S16–S19 的课件整理；标注“编者推导”“编者解释”的内容用于补足教材未写出的逐步代数过程。原页图中细节尚待完整视觉转录，因此本版**不声称已逐页核对页码**，也不编造新的具体页码事实；涉及“课堂作业”“应用图片”等仅保留其来源定位。

---

## 10.1 为什么多层网络需要非线性 {#ml-10-01}

### 直观直觉：一个最简单的反例——XOR

第三章的线性分类器（感知机、Fisher、logistic）学习的是“一条直线（或超平面）把空间一分为二”。但有一类问题，**任何一条直线都分不开**，最经典的例子就是异或（XOR）：

| 输入 $(x_1,x_2)$ | 标签 $y$ |
| --- | --- |
| $(0,0)$ | 0 |
| $(0,1)$ | 1 |
| $(1,0)$ | 1 |
| $(1,1)$ | 0 |

如果你在坐标纸上点出这四点，会发现“标签为 1 的点”被“标签为 0 的点”夹在中间，没有任何一条直线能把红蓝分开。单层线性模型必然失败。

那么“多叠几层线性层”行不行？**不行，因为层与层串起来仍然是线性变换**。下面用代数把这个结论讲死。

### 过程讲解：两层线性串联仍然线性

**假设与符号**（与全书统一符号一致；列向量记法）

- 第 $l$ 层的宽度为 $d_l$（即该层神经元个数，也是该层激活向量的维度）。
- 输入记作 $\mathbf a^{(0)}=\mathbf x\in\mathbb R^{d_0=\mathbf d}$。
- 第 $l$ 层先做仿射变换，再做逐元素非线性激活：

$$\mathbf z^{(l)}=W^{(l)}\mathbf a^{(l-1)}+\mathbf b^{(l)},\qquad \mathbf a^{(l)}=\phi_l\!\left(\mathbf z^{(l)}\right).$$

**维度约定（务必先确认）**

$$W^{(l)}\in\mathbb R^{d_l\times d_{l-1}},\qquad \mathbf b^{(l)},\mathbf z^{(l)},\mathbf a^{(l)}\in\mathbb R^{d_l}.$$

也就是说 $W^{(l)}$ 的每一行对应一个输出神经元，每一列对应一个输入神经元；偏置 $\mathbf b^{(l)}$ 与输出同维，对每个输出神经元各加一个标量。

**起始式**：考虑连续两层“中间没有非线性”的仿射变换（即把第一层的输出 $\mathbf a^{(1)}$ 直接当作第二层的输入，跳过激活）：

$$\mathbf z^{(2)}=W^{(2)}\big(W^{(1)}\mathbf a^{(0)}+\mathbf b^{(1)}\big)+\mathbf b^{(2)}.$$

**逐步推导**

1. 先把 $W^{(2)}$ 分配进括号（矩阵对向量加法的分配律）：
   $$\mathbf z^{(2)}=W^{(2)}W^{(1)}\mathbf a^{(0)}+W^{(2)}\mathbf b^{(1)}+\mathbf b^{(2)}.$$
2. 令组合权重 $\widetilde W=W^{(2)}W^{(1)}$，组合偏置 $\widetilde{\mathbf b}=W^{(2)}\mathbf b^{(1)}+\mathbf b^{(2)}$，则
   $$\mathbf z^{(2)}=\widetilde W\,\mathbf a^{(0)}+\widetilde{\mathbf b}.$$

**结论**：两层线性（仿射）变换的复合，仍然是一个仿射变换 $\widetilde W\in\mathbb R^{d_2\times d_0}$、$\widetilde{\mathbf b}\in\mathbb R^{d_2}$。推而广之，无论叠多少层纯线性层，整体等价于**单个**线性层。

**成立条件 / 边界**：该结论依赖“两层之间不插入非线性”。一旦在每层之后插入非线性激活 $\phi_l$，上面的合并就不再成立——这正是我们引入非线性的原因。

> **记住**：多层“表示能力”的提升，来源不是“多”，而是“非线性”。没有非线性，深和浅等价。

### 常见激活函数（编者解释）

| 名称 | 公式 | 特点与注意 |
| --- | --- | --- |
| Sigmoid | $\sigma(z)=\dfrac{1}{1+e^{-z}}$ | 输出 $(0,1)$，两端饱和导数趋于 0，易引发梯度衰减 |
| Tanh | $\tanh(z)=\dfrac{e^z-e^{-z}}{e^z+e^{-z}}$ | 输出 $(-1,1)$，零中心化，同样存在饱和 |
| ReLU | $\phi(z)=\max(0,z)$ | 正区间导数恒为 1，缓解梯度衰减；但负区间恒为 0，可能产生“长期不激活的死单元” |

最后一层按任务选择：回归用实值输出、二分类用 sigmoid（配二元交叉熵）、多分类用 softmax（配交叉熵）。

### 过程讲解：用手工权重构造 XOR（保留例题并核对数字）

**编者自编 XOR 构造**（仅证明“表示能力存在”，不证明梯度训练一定找到它）：

$$h_1=\max(0,x_1+x_2),\qquad h_2=\max(0,x_1+x_2-1),\qquad \hat y=h_1-2h_2.$$

**逐点核对**（把四个二元输入代进去，确认得到 0,1,1,0）

- $(x_1,x_2)=(0,0)$：$x_1+x_2=0\Rightarrow h_1=0$；$x_1+x_2-1=-1\Rightarrow h_2=0$；$\hat y=0-0=0$。✓
- $(0,1)$：$x_1+x_2=1\Rightarrow h_1=1$；$x_1+x_2-1=0\Rightarrow h_2=0$；$\hat y=1-0=1$。✓
- $(1,0)$：对称地 $x_1+x_2=1\Rightarrow h_1=1,h_2=0$；$\hat y=1$。✓
- $(1,1)$：$x_1+x_2=2\Rightarrow h_1=2$；$x_1+x_2-1=1\Rightarrow h_2=1$；$\hat y=2-2\times1=0$。✓

得到序列 $0,1,1,0$，与 XOR 标签一致。这里第一隐藏单元 $h_1$ 在“至少一位为 1”时开启，第二隐藏单元 $h_2$ 只在“两位都为 1”时开启，二者之差恰好区分了“恰好一个 1”和“两个都是 1 / 两个都是 0”。

> **常见误区**：手工能构造出一组权重，只说明网络“容量上”能表达 XOR，不等于训练算法一定能收敛到这组权重。表示能力 ≠ 可训练性。

---

## 10.2 BP 是高效组织链式法则 {#ml-10-02}

### 直观直觉：为什么需要“反向传播”

网络有上百万参数，我们最终要算的是损失 $J$ 对每个参数的偏导数 $\partial J/\partial W^{(l)}$、$\partial J/\partial \mathbf b^{(l)}$。直接对每个参数单独求导不现实。反向传播（Backpropagation, BP）的本质就是：**用一张计算图，按链式法则从输出往输入方向，把梯度“高效、复用”地算一遍**。

- 前向：一层层算 $\mathbf z^{(l)},\mathbf a^{(l)}$ 并**保存中间结果**（反向要用）。
- 反向：从输出层往回，利用“每个节点的局部梯度”乘上“已算好的上游梯度”，一路复用，避免重复计算。

本章的反向传播只为参数**提供梯度**；真正用梯度更新参数的是梯度下降，其思想在[第三章的感知机与梯度更新](03-linear.md#ml-03-04)中已经出现——第三章的“错了朝正确方向移动”本质就是沿梯度（或次梯度）下降。

### 记号与误差信号定义

- 单样本标量损失记为 $J$（批量时再对样本求和或平均，须与损失定义一致）。
- 定义第 $l$ 层的**局部误差信号**（对“预激活”$\mathbf z^{(l)}$ 的梯度）：

$$\boldsymbol\delta^{(l)}\;=\;\frac{\partial J}{\partial \mathbf z^{(l)}}\;\in\;\mathbb R^{d_l}.$$

$\boldsymbol\delta^{(l)}$ 的物理意义：第 $l$ 层每个神经元“预激活”变动一点点，会对总损失造成多大影响。它是连接“上游梯度”和“本层参数梯度”的桥梁。

### 过程讲解（核心）：隐藏层误差信号递推 $\boldsymbol\delta^{(l)}$

**目标**：用下一层误差 $\boldsymbol\delta^{(l+1)}$ 表达本层误差 $\boldsymbol\delta^{(l)}$，得到反向递推式。

**依赖关系（先看计算图）**：本层预激活 $\mathbf z^{(l)}$ 经激活得到 $\mathbf a^{(l)}=\phi_l(\mathbf z^{(l)})$；再进入下一层 $\mathbf z^{(l+1)}=W^{(l+1)}\mathbf a^{(l)}+\mathbf b^{(l+1)}$。所以 $J$ 通过 $\mathbf a^{(l)}$ 间接依赖 $\mathbf z^{(l)}$。

**起始式（对第 $i$ 个分量写链式法则）**：

$$\delta^{(l)}_i\;=\;\frac{\partial J}{\partial z^{(l)}_i}\;=\;\sum_j \underbrace{\frac{\partial J}{\partial z^{(l+1)}_j}}_{\delta^{(l+1)}_j}\;\frac{\partial z^{(l+1)}_j}{\partial z^{(l)}_i}.$$

**关键一步：算 $\partial z^{(l+1)}_j/\partial z^{(l)}_i$**。展开下一层预激活的第 $j$ 个分量：

$$z^{(l+1)}_j\;=\;\sum_k W^{(l+1)}_{jk}\,a^{(l)}_k+b^{(l+1)}_j\;=\;\sum_k W^{(l+1)}_{jk}\,\phi_l\!\left(z^{(l)}_k\right)+b^{(l+1)}_j.$$

对 $z^{(l)}_i$ 求偏导时，只有 $k=i$ 那一项含有 $z^{(l)}_i$，其余项导数为 0：

$$\frac{\partial z^{(l+1)}_j}{\partial z^{(l)}_i}\;=\;W^{(l+1)}_{ji}\,\phi_l'\!\left(z^{(l)}_i\right).$$

**代回起始式并整理**：

$$\delta^{(l)}_i\;=\;\sum_j \delta^{(l+1)}_j\,W^{(l+1)}_{ji}\,\phi_l'\!\left(z^{(l)}_i\right)\;=\;\left[\big(W^{(l+1)}\big)^{\!T}\boldsymbol\delta^{(l+1)}\right]_i\;\phi_l'\!\left(z^{(l)}_i\right).$$

写成向量形式（逐元素乘法用 $\odot$）：

$$\boxed{\;\boldsymbol\delta^{(l)}=\big[\big(W^{(l+1)}\big)^{\!T}\boldsymbol\delta^{(l+1)}\big]\odot\phi_l'\!\left(\mathbf z^{(l)}\right)\;}$$

**维度检查**：
- $W^{(l+1)}\in\mathbb R^{d_{l+1}\times d_l}\Rightarrow\big(W^{(l+1)}\big)^{\!T}\in\mathbb R^{d_l\times d_{l+1}}$；
- $\boldsymbol\delta^{(l+1)}\in\mathbb R^{d_{l+1}}$；
- 二者相乘 $\mathbb R^{d_l\times d_{l+1}}\times\mathbb R^{d_{l+1}}\to\mathbb R^{d_l}$；
- $\phi_l'(\mathbf z^{(l)})\in\mathbb R^{d_l}$，逐元素乘后仍 $\mathbb R^{d_l}$，与 $\boldsymbol\delta^{(l)}$ 维度一致。✓

> **记住**：矩阵转置 $(W^{(l+1)})^{\!T}$ 干的事情是“把下一层每个单元对本层第 $i$ 个神经元的贡献汇总起来”；$\odot$ 完成的是激活函数的链式法则因子。

### 过程讲解：参数梯度 $\partial J/\partial W^{(l)}$ 与维度检查

**目标**：把 $\boldsymbol\delta^{(l)}$ 变成可直接用于更新的权重梯度。

**起始式**：看 $W^{(l)}$ 第 $j$ 行第 $k$ 列元素如何影响损失。由前向式

$$z^{(l)}_j=\sum_k W^{(l)}_{jk}\,a^{(l-1)}_k+b^{(l)}_j,$$

所以 $\partial z^{(l)}_j/\partial W^{(l)}_{jk}=a^{(l-1)}_k$，与 $j$ 行其它列无关。

**链式法则**：

$$\frac{\partial J}{\partial W^{(l)}_{jk}}\;=\;\sum_i \frac{\partial J}{\partial z^{(l)}_i}\,\frac{\partial z^{(l)}_i}{\partial W^{(l)}_{jk}}\;=\;\delta^{(l)}_j\,a^{(l-1)}_k.$$

（只有 $i=j$ 的项非零，因为 $W^{(l)}_{jk}$ 只影响 $z^{(l)}_j$。）

**写成矩阵**：把所有 $(j,k)$ 拼成矩阵，正好是“列向量 $\boldsymbol\delta^{(l)}$ 乘行向量 $(\mathbf a^{(l-1)})^{\!T}$”：

$$\boxed{\;\frac{\partial J}{\partial W^{(l)}}=\boldsymbol\delta^{(l)}\big(\mathbf a^{(l-1)}\big)^{\!T},\qquad \frac{\partial J}{\partial \mathbf b^{(l)}}=\boldsymbol\delta^{(l)}\;}$$

**维度检查**（题目明确要求核对）：
- $\boldsymbol\delta^{(l)}\in\mathbb R^{d_l\times 1}$；
- $(\mathbf a^{(l-1)})^{\!T}\in\mathbb R^{1\times d_{l-1}}$；
- 外积结果 $\mathbb R^{d_l\times 1}\times\mathbb R^{1\times d_{l-1}}\to\mathbb R^{d_l\times d_{l-1}}$，恰好等于 $W^{(l)}$ 的形状。✓

> **常见误区**：有人写成 $\boldsymbol\delta^{(l)}(\mathbf a^{(l-1)})^{\!T}$ 却搞反内外积，得到 $d_{l-1}\times d_l$ 的错误形状。请始终用“行数=输出维、列数=输入维”回扣 $W^{(l)}$ 的维度。

### 过程讲解：输出层 softmax + 交叉熵的简化误差 $\boldsymbol\delta^{(L)}=\mathbf p-\mathbf y$

这是 BP 在分类任务里**最省事也最容易算错**的一步。设输出层（第 $L$ 层）有 $K$ 类：

- 预激活（logits）$\mathbf z^{(L)}\in\mathbb R^K$；
- softmax 概率 $\displaystyle p_i=\frac{e^{z^{(L)}_i}}{\sum_{j=1}^K e^{z^{(L)}_j}}$；
- one-hot 标签 $\mathbf y\in\mathbb R^K$，$\sum_i y_i=1$；
- 交叉熵损失 $\displaystyle J=-\sum_{i=1}^K y_i\log p_i$。

**目标**：求 $\delta^{(L)}_i=\partial J/\partial z^{(L)}_i$。

**第一步：展开 $\partial J/\partial z^{(L)}_i$**

$$\frac{\partial J}{\partial z^{(L)}_i}\;=\;-\sum_{k=1}^K y_k\,\frac{\partial \log p_k}{\partial z^{(L)}_i}.$$

**第二步：算 $\partial\log p_k/\partial z^{(L)}_i$**。先写出 $\log p_k=z^{(L)}_k-\log\!\big(\sum_j e^{z^{(L)}_j}\big)$，再求导：

$$\frac{\partial \log p_k}{\partial z^{(L)}_i}\;=\;\frac{\partial z^{(L)}_k}{\partial z^{(L)}_i}-\frac{e^{z^{(L)}_i}}{\sum_j e^{z^{(L)}_j}}\;=\;\delta_{ki}-p_i,$$

其中 $\delta_{ki}$ 是克罗内克符号（$k=i$ 时为 1，否则为 0）。

**第三步：代回并化简**

$$\frac{\partial J}{\partial z^{(L)}_i}\;=\;-\sum_{k=1}^K y_k(\delta_{ki}-p_i)\;=\;-y_i+\underbrace{p_i\sum_{k=1}^K y_k}_{=p_i\cdot 1}\;=\;p_i-y_i.$$

于是

$$\boxed{\;\boldsymbol\delta^{(L)}=\mathbf p-\mathbf y\;}$$

**结论与边界**：该简化**仅在“softmax 输出 + 交叉熵损失 + one-hot 标签”三者同时成立**时才有效；若输出层用别的激活（如 sigmoid 配二元交叉熵、或 MSE），$\boldsymbol\delta^{(L)}$ 会是不同形式。它的维度 $\mathbb R^K$，与类别数一致。

> **小数值检查**：假设三分类，$p=(0.2,0.5,0.3)$，真实类是第 2 类 $y=(0,1,0)$，则 $\delta^{(L)}=(0.2,-0.5,0.3)$。注意正确类别处为负（降低其 logit 会让损失上升，所以梯度要“推高”它，负号即对应上升方向），符合直觉。

### 批量、实现与训练注意事项

- **批量训练**：对 $N$ 个样本的损失 $J=\frac1N\sum_{n=1}^N J^{(n)}$（或求和），梯度同样对样本取平均（或求和），须与损失定义保持一致。
- **前向要存、反向要逆序**：必须保存各层 $\mathbf a^{(l)},\mathbf z^{(l)}$ 供反向使用；反向按依赖逆序计算。
- **统一更新**：应在**全部梯度算好后再统一更新**参数，避免同一轮反向传播混用“已更新”和“未更新”的权重——否则 BP 的链式法则假设（图结构固定）被破坏。
- **BP 与梯度下降是两件事**：BP 负责“算梯度”，梯度下降（或 Adam 等）负责“用梯度更新”。不要把二者混为一谈。

### 梯度消失 / 爆炸（保留并深化）

深层网络里梯度是许多因子连乘：$\boldsymbol\delta^{(l)}$ 含 $(W^{\!T})^{L-l}$ 与一堆 $\phi'$。若这些因子绝对值普遍小于 1（如饱和 sigmoid，$\phi'\in(0,0.25]$），乘积指数级趋零——**梯度消失**；若普遍大于 1，则**梯度爆炸**。ReLU 在正区间导数恒为 1，能缓解部分消失，但负区间导数 0 会产生“长期不激活的死单元”。因此初始化、归一化（Batch/Layer Norm）、残差连接与优化器设置都会影响训练，**并非“换个激活就一定能解决”**。

---

## 10.3 CNN：局部连接和权重共享 {#ml-10-03}

### 直观直觉：为什么卷积“省参数又有效”

自然图像里，“边缘、角点”这类局部模式可能在图像任何位置出现。若每个位置都学一套独立检测器，参数会爆炸且泛化差。卷积层让**同一组核参数在不同空间位置复用**（权重共享），一个输出通道对应一种局部模式；输入的不同通道（RGB）在求和时共享同一组核权重，因此参数不随图像空间尺寸增长。

### 过程讲解：输出尺寸公式 $L_{out}=\left\lfloor\frac{L_{in}+2P-F}{S}\right\rfloor+1$

**假设与符号（一维，便于看清；二维对高、宽分别计算）**

- $L_{in}$：输入长度（如宽度方向的像素数）；
- $F$：卷积核大小（filter size）；
- $P$：单侧填充（padding），左右各加 $P$，故填充后长度为 $L_{in}+2P$；
- $S>0$：步长（stride），每次核滑动 $S$ 个像素；
- 要求核在填充后的输入上**至少能完整放置一次**，否则该层定义不成立。

**起始式（数“合法起始位置”）**：核第一次放在位置 0，覆盖 $[0,F-1]$。设核左端起始位置为 $s$，则需满足

$$s+F-1\le (L_{in}+2P)-1\quad\Longrightarrow\quad s\le L_{in}+2P-F.$$

合法的 $s$ 为 $0,S,2S,\dots$，不超过上界，其个数为

$$\text{个数}=\left\lfloor\frac{L_{in}+2P-F}{S}\right\rfloor+1.$$

**结论（这也是二维尺寸的逐轴公式）**：

$$\boxed{\;L_{out}=\left\lfloor\frac{L_{in}+2P-F}{S}\right\rfloor+1\;}$$

**二维**：对高 $H$、宽 $W$ 分别套用上式，得到 $(H_{out},W_{out})$，通道数变为 $C_{out}$。

**小数值检查（无填充）**：原例题 $L_{in}=32,F=5,P=0,S=1$：

$$L_{out}=\left\lfloor\frac{32+0-5}{1}\right\rfloor+1=27+1=28.\quad\checkmark$$

**补充数值例子（同尺寸卷积）**：若 $L_{in}=7,F=3,P=1,S=1$，则 $L_{out}=\lfloor(7+2-3)/1\rfloor+1=6+1=7$，输出与输入等长——这正是对图像做“same padding”的常见情形。

### 过程讲解：参数量公式（保留例题并核对 456）

**假设与符号**

- 输入通道 $C_{in}$，输出通道 $C_{out}$；
- 每个输出通道对应一个大小为 $F_h\times F_w$ 的空域核，且该核要**跨全部 $C_{in}$ 个输入通道求和**；
- 每个输出通道另有一个标量偏置。

**起始式（数一个输出通道的参数）**：一个输出通道的核权重数为

$$F_h\times F_w\times C_{in},$$

再加 1 个偏置，共 $F_hF_wC_{in}+1$。

**对 $C_{out}$ 个通道求和**：

$$\boxed{\;\text{参数量}=C_{out}\big(F_hF_w\,C_{in}+1\big)\;}$$

**关键点**：该式**不含 $L_{in}$ 或图像空间尺寸**——因为权重在空域共享，图像变大参数不变（这是 CNN 相对全连接的核心优势）。

**核对原例题**：$32\times32\times3$ 输入，$C_{in}=3$，6 个 $5\times5$ 核即 $C_{out}=6,F_h=F_w=5$，步长 1、不填充：

$$\text{参数量}=6\big(5\times5\times3+1\big)=6(75+1)=6\times76=456.\quad\checkmark$$

输出为 $28\times28\times6$（由上方尺寸公式得 $28\times28$，通道 $C_{out}=6$）。

### 池化、平移等变性与边界说明

- **最大/平均池化**通常**没有可学习参数**，作用是降采样；但它会丢失精确位置信息（只保留“某区域内最大值”）。
- **局部共享带来平移等变性**：输入平移，输出按同样方式平移（equivariance），即“卷积 + 平移 = 平移 + 卷积”；但 CNN **不自动保证任意平移下分类结果完全不变**（invariance）——真正的不变量通常要靠池化/全局池化与训练数据共同塑造。不要把“等变”误说成“不变”。

---

## 10.4 RNN：共享参数的时间展开 {#ml-10-04}

### 直观直觉：序列依赖过去

很多任务（语言、时序）中，当前预测依赖历史。若用前馈网络硬接固定长度历史，窗口长度受限。循环神经网络（RNN）让**同一组参数在不同时间步复用**，从而用有限参数处理任意长度序列。代价是：训练时要沿时间“展开”成一张很深的图，梯度沿时间回传。

### 过程讲解：时间展开与误差信号 $\boldsymbol\delta_t$

**假设与符号**

- 隐状态 $\mathbf h_t\in\mathbb R^{d_h}$，输入 $\mathbf x_t\in\mathbb R^{d_x}$，输出 $\mathbf o_t\in\mathbb R^{d_o}$；
- 预激活（隐藏层“激活前”的值）记为 $\mathbf u_t=W_h\mathbf h_{t-1}+W_x\mathbf x_t+\mathbf b$，则 $\mathbf h_t=\phi(\mathbf u_t)$；
- 各时刻**共享**同一组 $W_h,W_x,W_o,\mathbf b,\mathbf c$；
- 总损失 $J=\sum_{t=1}^T \ell_t$，设 $\boldsymbol\delta_t=\partial J/\partial\mathbf u_t\in\mathbb R^{d_h}$，输出梯度 $\mathbf e_t=\partial\ell_t/\partial\mathbf o_t\in\mathbb R^{d_o}$。

**展开视角**：把 $\mathbf h_1\to\mathbf h_2\to\cdots\to\mathbf h_T$ 在纸上横向铺开，每个时间步是图中的一个“层”，共享参数跨层复用——所以 RNN 的训练图在“时间维”上非常深。

**起始式（对 $t<T$ 写 $\boldsymbol\delta_t$）**：$\mathbf u_t$ 影响损失有两条路——(1) 经 $\mathbf h_t$ 直接到 $\mathbf o_t$（输出损失 $\ell_t$）；(2) 经 $\mathbf h_t$ 到下一步预激活 $\mathbf u_{t+1}$（从而影响 $\ell_{t+1},\dots$）。因此

$$\boldsymbol\delta_t=\frac{\partial J}{\partial \mathbf u_t}=\underbrace{\frac{\partial J}{\partial \mathbf h_t}}_{\text{来自两条路}}\odot\phi'(\mathbf u_t).$$

而 $\partial J/\partial \mathbf h_t$ 由两部分组成：

$$\frac{\partial J}{\partial \mathbf h_t}=\big(W_o\big)^{\!T}\mathbf e_t\;+\;\big(W_h\big)^{\!T}\boldsymbol\delta_{t+1}.$$

理由：$\mathbf h_t$ 进入输出 $\mathbf o_t=W_o\mathbf h_t+\mathbf c$（带来 $W_o^{\!T}\mathbf e_t$），也进入下一步 $\mathbf u_{t+1}=W_h\mathbf h_t+\cdots$（带来 $W_h^{\!T}\boldsymbol\delta_{t+1}$）。

**合并得到递推式**（末态之后无未来，$t=T$ 时只有输出项，$\boldsymbol\delta_{T+1}=0$）：

$$\boxed{\;\boldsymbol\delta_t=\big(W_o^{\!T}\mathbf e_t+W_h^{\!T}\boldsymbol\delta_{t+1}\big)\odot\phi'(\mathbf u_t),\qquad \boldsymbol\delta_{T+1}=0\;}$$

**维度检查**：
- $W_o^{\!T}\in\mathbb R^{d_h\times d_o}$，$\mathbf e_t\in\mathbb R^{d_o}\Rightarrow$ 乘积 $\mathbb R^{d_h}$；
- $W_h^{\!T}\in\mathbb R^{d_h\times d_h}$，$\boldsymbol\delta_{t+1}\in\mathbb R^{d_h}\Rightarrow$ 乘积 $\mathbb R^{d_h}$；
- $\phi'(\mathbf u_t)\in\mathbb R^{d_h}$，逐元素乘后仍为 $\mathbb R^{d_h}=\boldsymbol\delta_t$ 的维度。✓

### 过程讲解：共享参数的梯度必须“按时间求和”

**目标**：因为 $W_h$ 在每个时间步都出现，它对总损失的梯度是**各时间步贡献之和**，不能只取最后一步。

**起始式**：对每个时间步 $t$，$W_h$ 通过 $\mathbf u_t=W_h\mathbf h_{t-1}+\cdots$ 影响损失。类似 10.2 的权重梯度推导：

$$\frac{\partial \ell_{\le T}}{\partial W_h}\;\supset\; \frac{\partial J}{\partial \mathbf u_t}\frac{\partial \mathbf u_t}{\partial W_h}=\boldsymbol\delta_t\,\mathbf h_{t-1}^{\!T},\qquad \frac{\partial \mathbf u_t}{\partial W_h}=\mathbf h_{t-1}^{\!T}.$$

**对所有时间步求和**（共享参数把各步梯度累加）：

$$\boxed{\;\frac{\partial J}{\partial W_h}=\sum_{t=1}^{T}\boldsymbol\delta_t\,\mathbf h_{t-1}^{\!T}\;}$$

**维度检查**：$\boldsymbol\delta_t\in\mathbb R^{d_h}$，$\mathbf h_{t-1}\in\mathbb R^{d_h}\Rightarrow\boldsymbol\delta_t\mathbf h_{t-1}^{\!T}\in\mathbb R^{d_h\times d_h}$，与 $W_h$ 形状一致，求和后仍为 $\mathbb R^{d_h\times d_h}$。✓

**同类结果（一并给出，便于实现）**：

$$\frac{\partial J}{\partial W_x}=\sum_t \boldsymbol\delta_t\,\mathbf x_t^{\!T},\qquad \frac{\partial J}{\partial W_o}=\sum_t \mathbf e_t\,\mathbf h_t^{\!T},\qquad \frac{\partial J}{\partial \mathbf b}=\sum_t\boldsymbol\delta_t,\qquad \frac{\partial J}{\partial \mathbf c}=\sum_t\mathbf e_t.$$

### 截断 BPTT（必须说明的边界）

**截断 BPTT（Truncated BPTT）**只把梯度回传有限时间窗（例如只看前 $h$ 步），以换取计算与内存的可控。但这是**近似**：它改变了梯度估计，并不是完整 BPTT 的精确结果。课件 S18 21 页明确列有“截断深度为 $h$ 的学习（课堂作业）”，本版保存其来源；原页图中细节尚待完整视觉转录，故不在此展开其具体数值设定。

---

## 10.5 注意力、Transformer 与预训练 {#ml-10-05}

### 直观直觉：固定长度状态装不下长序列

RNN 把历史压缩进一个定长隐状态 $\mathbf h_t$，长输入时“远处信息”容易被冲掉。注意力机制换一种思路：当前位置（查询）直接去“翻看”序列里所有位置（键—值），按**相关性**加权取用，需要多少就看多少。

### 过程讲解：缩放点积注意力逐步写清

**假设与符号**

- 查询 $Q\in\mathbb R^{n_q\times d_k}$，键 $K\in\mathbb R^{n_k\times d_k}$，值 $V\in\mathbb R^{n_k\times d_v}$；
- 掩码 $M\in\mathbb R^{n_q\times n_k}$，允许位置为 $0$、禁止位置为 $-\infty$；
- softmax 沿**每个查询行、对键维度 $n_k$** 归一化。

**标准公式**：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\!T}}{\sqrt{d_k}}+M\right)V.$$

下面把这一步拆成 5 个小步，每一步都标注维度与含义：

1. **算相关性分数** $S=QK^{\!T}$。
   - 维度：$\mathbb R^{n_q\times d_k}\times\mathbb R^{d_k\times n_k}\to\mathbb R^{n_q\times n_k}$；
   - 含义：$S_{ij}=Q_{i:}\cdot K_{j:}$，即“查询 $i$ 与键 $j$ 的点积相似度”。

2. **缩放除以 $\sqrt{d_k}$**。
   - 含义：点积的方差随维度 $d_k$ 增大而增大，数值过大使 softmax 进入“饱和区”、梯度趋零；除以 $\sqrt{d_k}$ 把分数方差稳定在量级 1 附近，训练更稳。

3. **加掩码 $+M$**。
   - 禁止位置加 $-\infty$，使该位置在后续 softmax 中权重趋近于 0；允许位置加 0，不变。

4. **按行 softmax**。
   - 对每行 $i$： $\displaystyle A_{ij}=\frac{\exp\!\big(S_{ij}/\sqrt{d_k}+M_{ij}\big)}{\sum_{j'=1}^{n_k}\exp\!\big(S_{ij'}/\sqrt{d_k}+M_{ij'}\big)}$；
   - 每行之和为 1，得到“权重分布” $A\in\mathbb R^{n_q\times n_k}$。

5. **用权重对值加权求和** $O=AV$。
   - 维度：$\mathbb R^{n_q\times n_k}\times\mathbb R^{n_k\times d_v}\to\mathbb R^{n_q\times d_v}$；
   - 含义：$O_{i:}=\sum_j A_{ij}V_{j:}$，即“查询 $i$ 的输出是各值向量的相关加权组合”。

> **小数值检查（无掩码、单查询）**：设 $d_k=2$，$Q=[1,0]$，$K=[[1,0],[0,1],[-1,0]]$，$V=[[10],[20],[30]]$。
> - 分数 $S=QK^{\!T}=[1,0,-1]$；缩放（这里 $d_k=2$，$\sqrt2\approx1.414$）得约 $[0.707,0,-0.707]$；
> - softmax：$e^{0.707}\approx2.028,\ e^0=1,\ e^{-0.707}\approx0.493$，和 $\approx3.521$，权重 $\approx[0.576,0.284,0.140]$；
> - 输出 $O\approx0.576\times10+0.284\times20+0.140\times30\approx5.76+5.68+4.20=15.64$。
> 注意最相似的键（第一个）拿到了最大权重，输出偏向其对应值 10——符合“按相关性取用”的直觉。

### 掩码的两个关键边界（必须说明）

- **因果掩码（causal mask）**：自回归训练时，位置 $i$ 只能看 $j\le i$。实现上令 $M_{ij}=0$（当 $j\le i$）、$M_{ij}=-\infty$（当 $j>i$），防止“偷看未来”。
- **绝不能整行全遮**：若某个查询行的所有键都被设为 $-\infty$，则该行 softmax 是 $\exp(-\infty)/\sum\exp(-\infty)=0/0$，结果为 **NaN/未定义**。实现时必须保证**每个查询行至少有一个允许位置**（例如因果掩码下，第 1 个位置至少能看自己）。这是注意力实现里最易踩的坑。

### 多头、位置编码与预训练（保留并深化）

- **多头注意力**：把 $Q,K,V$ 线性投影到多个子空间，各自独立做上面的 5 步再拼接，让模型在不同表示子空间分别建立关系。
- **位置编码**补回顺序信息（注意力本身对输入排列不敏感）；**残差连接 + 层归一化**帮助信息与前向/反向梯度稳定传播。
- **预训练目标应分别说明**：掩码语言模型（MLM）根据上下文恢复被遮符号；自回归语言模型把序列概率分解为
  $$p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).$$
  微调目标与下游评估须与预训练目标区分开。
- 课件介绍 ELMo、BERT、GPT、编码器—解码器与 RLHF。本书只保留稳定机制说明，不把某个版本的参数量或排名写成通用事实。RLHF 等人工反馈对齐通常含监督微调、偏好/奖励建模、策略优化等环节，不能简化为“模型知道了人的真实意图”。

---

## 10.6 生成模型与安全边界 {#ml-10-06}

### GAN：极小极大博弈（保留目标并解释）

GAN 同时训练生成器 $G$ 与鉴别器 $D$，经典目标为

$$\min_{G}\max_{D}\; \mathbb E_{x\sim p_{data}}\big[\log D(x)\big]+\mathbb E_{z\sim p(z)}\big[\log\big(1-D(G(z))\big)\big].$$

- **鉴别器 $D$** 尽量把真实样本判为 1、生成样本判为 0（最大化上式）；
- **生成器 $G$** 尽量骗过 $D$（最小化上式中第二项）。
- 这是**二人零和博弈**，交替优化。它**可能不稳定、出现模式坍塌**（生成样本多样性不足），不能保证有限训练必然达到理想均衡。不要把 GAN 描述成“一定能收敛到完美生成器”。

### 扩散模型：闭式加噪（保留公式并解释）

扩散模型的前向过程逐步加噪。常用闭式写法

$$\mathbf x_t=\sqrt{\bar\alpha_t}\,\mathbf x_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\qquad \boldsymbol\epsilon\sim\mathcal N(0,I),$$

其中 $\bar\alpha_t$ 为各步保真系数之积（随时间递减，噪声占比递增）。

- **含义**：$\mathbf x_t$ 是“原图 $\mathbf x_0$ 按 $\sqrt{\bar\alpha_t}$ 缩放 + 标准高斯噪声按 $\sqrt{1-\bar\alpha_t}$ 缩放”的线性组合；当 $\bar\alpha_t\to0$ 时 $\mathbf x_t$ 趋近于纯噪声。
- **训练**：网络学习“给定时间 $t$ 和带噪样本 $\mathbf x_t$ 时预测噪声 $\boldsymbol\epsilon$”。
- **采样**：按所选反向过程逐步去噪，产生数据。
- **边界（务必说明）**：生成的是**学习分布中的新样本**，不保证还原某张特定训练图；也不能忽略时间相关系数 $\bar\alpha_t$、而反复直接减同一个噪声——不同 $t$ 的噪声尺度不同，必须按对应系数处理。

### 对抗样本与安全边界（保留概念定位）

课件末尾讨论对抗样本：输入在给定扰动预算（如 $\ell_\infty$ 范数界）内做微小变化，可能导致错误预测。应区分白盒、基于分数、基于决策与迁移等威胁模型，并报告扰动范数、预算与评价条件。这里是课程概念整理，**不包含对外部系统的攻击实验**。

> **来源定位（保留）**：自组织映射、RBF 网络、RBM、强化学习网络等在 S16 37–58 页概述性出现，本版记录其定位，但不宣称已完整推导这些扩展算法。S19 中应用图片与早期平台列表保留在来源材料中，尚未逐项写成经过核实的现状说明。

---

## 10.7 小结、常见误区与练习 {#ml-10-07}

### 一页速查（本章核心公式）

| 主题 | 公式 | 维度要点 |
| --- | --- | --- |
| 前向一层 | $\mathbf z^{(l)}=W^{(l)}\mathbf a^{(l-1)}+\mathbf b^{(l)},\ \mathbf a^{(l)}=\phi_l(\mathbf z^{(l)})$ | $W^{(l)}\!\in\!\mathbb R^{d_l\times d_{l-1}}$，$\mathbf a^{(l)}\!\in\!\mathbb R^{d_l}$ |
| 隐藏层误差 | $\boldsymbol\delta^{(l)}=\big[(W^{(l+1)})^{\!T}\boldsymbol\delta^{(l+1)}\big]\odot\phi_l'(\mathbf z^{(l)})$ | 输出 $\mathbb R^{d_l}$ |
| 权重梯度 | $\partial J/\partial W^{(l)}=\boldsymbol\delta^{(l)}(\mathbf a^{(l-1)})^{\!T}$ | 外积得 $\mathbb R^{d_l\times d_{l-1}}$ |
| 输出误差（softmax+CE） | $\boldsymbol\delta^{(L)}=\mathbf p-\mathbf y$ | $\mathbb R^K$，需 one-hot |
| CNN 输出尺寸 | $L_{out}=\lfloor(L_{in}+2P-F)/S\rfloor+1$ | 二维对高宽各算一次 |
| CNN 参数量 | $C_{out}(F_hF_wC_{in}+1)$ | 不含空间尺寸 |
| RNN 误差 | $\boldsymbol\delta_t=(W_o^{\!T}\mathbf e_t+W_h^{\!T}\boldsymbol\delta_{t+1})\odot\phi'(\mathbf u_t)$ | 末态 $\boldsymbol\delta_{T+1}=0$ |
| RNN 共享梯度 | $\partial J/\partial W_h=\sum_t\boldsymbol\delta_t\mathbf h_{t-1}^{\!T}$ | 必须按时间求和 |
| 注意力 | $\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^{\!T}/\sqrt{d_k}+M)V$ | $Q\!\in\!\mathbb R^{n_q\times d_k},V\!\in\!\mathbb R^{n_k\times d_v}$ |
| 扩散闭式 | $\mathbf x_t=\sqrt{\bar\alpha_t}\mathbf x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$ | $\boldsymbol\epsilon\sim\mathcal N(0,I)$ |

### 常见误区汇总（编者总结）

1. **“多层 = 非线性”**：错。层间无激活时，多层线性等价于单层线性。
2. **“换 ReLU 就解决梯度消失”**：错。ReLU 缓解但不根除，且带来死单元。
3. **“卷积层参数随图像变大而变多”**：错。权重共享使参数量与空间尺寸无关。
4. **“CNN 平移不变”**：应为“平移等变”；不变性需池化/训练共同塑造。
5. **“截断 BPTT 是精确 BPTT”**：错。它是有限窗近似，改变了梯度估计。
6. **“注意力掩码可整行全遮”**：错。整行 $-\infty$ 使 softmax 未定义（NaN）。
7. **“GAN 必然收敛到完美生成器”**：错。可能不稳定、模式坍塌。
8. **“BP 就是训练”**：错。BP 只算梯度；更新由梯度下降（见[第三章](03-linear.md#ml-03-04)）完成。

### 练习与复算

本章涉及的例题数字（XOR 的 $0,1,1,0$、CNN 的 $28\times28\times6$ 与参数 $456$）建议亲手复算一遍，并对照[练习与可复现计算](../exercises.md)中对应条目。完成练习时先遮住解答，自行说明假设、定义符号、写出目标，再逐步计算——尤其要练习“先核对维度、再写公式”的习惯。
