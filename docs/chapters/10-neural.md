# 第十章 神经网络与深度学习：可学习表示的组合

> 往年资料：S16（概述），S17 4–47 页（多层感知机与 BP）、49–71 页（CNN），S18 3–27 页（RNN），S19 3–73 页（深度学习与前沿概览）。原始章节编号出现“第十一章／14.x”残留，本书按课程大纲统一为第十章。历史产品列表与性能数据未当作最新事实。

## 10.1 为什么多层网络需要非线性 {#ml-10-01}

第三章的单层线性分类器不能解决 XOR。把两个线性层串联却仍不能解决：$W_2(W_1\mathbf x+\mathbf b_1)+\mathbf b_2=(W_2W_1)\mathbf x+W_2\mathbf b_1+\mathbf b_2$，整体仍是线性加偏置。因此每层之间需要非线性激活。

令第 $l$ 层宽度为 $d_l$，$\mathbf a^{(0)}=\mathbf x$，

$$\mathbf z^{(l)}=W^{(l)}\mathbf a^{(l-1)}+\mathbf b^{(l)},\qquad\mathbf a^{(l)}=\phi_l(\mathbf z^{(l)}).$$

$W^{(l)}\in\mathbb R^{d_l\times d_{l-1}}$，$\mathbf b^{(l)},\mathbf z^{(l)},\mathbf a^{(l)}\in\mathbb R^{d_l}$。激活可用 sigmoid、tanh、ReLU。最后一层按任务选择实值输出、sigmoid 或 softmax，以及匹配的损失。

编者自编 XOR 构造：$h_1=\max(0,x_1+x_2)$，$h_2=\max(0,x_1+x_2-1)$，输出 $h_1-2h_2$。对二元输入 00、01、10、11 依次得到 0、1、1、0。这里手动构造权重只证明表示能力，不证明梯度训练一定找到它。

## 10.2 BP 是高效组织链式法则 {#ml-10-02}

设单样本标量损失为 $J$，局部误差信号定义为 $\boldsymbol\delta^{(l)}=\partial J/\partial\mathbf z^{(l)}$。对隐藏层，从下一层误差传回：

$$\boldsymbol\delta^{(l)}=[(W^{(l+1)})^T\boldsymbol\delta^{(l+1)}]\odot\phi_l'(\mathbf z^{(l)}).$$

矩阵转置把下一层每个单元对本层的贡献加起来；逐元素乘激活导数完成链式法则。参数梯度为

$$\frac{\partial J}{\partial W^{(l)}}=\boldsymbol\delta^{(l)}(\mathbf a^{(l-1)})^T,\qquad\frac{\partial J}{\partial\mathbf b^{(l)}}=\boldsymbol\delta^{(l)}.$$

维度检查：$d_l\times1$ 乘 $1\times d_{l-1}$ 正好得到权重矩阵尺寸。softmax 配合交叉熵时输出层 $\boldsymbol\delta^{(L)}=\mathbf p-\mathbf y$，其中 $\mathbf y$ 为 one-hot 标签。批量训练再对样本求和或求平均，须与损失定义一致。

前向计算保存中间激活，反向按依赖逆序计算梯度，再更新参数。应在全部梯度算好后统一更新，避免同一轮反向传播混用新旧权重。BP 求梯度，梯度下降用梯度更新，这两个步骤不同。

深层链式乘积可能导致梯度衰减或放大；sigmoid 饱和时导数小，ReLU 虽有助于部分情况，却可能产生长期不激活的单元。初始化、归一化、残差连接与优化设置都影响训练，并非“换激活就总能解决”。

## 10.3 CNN：局部连接和权重共享 {#ml-10-03}

图像中相同局部模式可能出现在许多位置，没必要每个位置都学习独立检测器。卷积层在不同空间位置使用同一组核参数，输出通道对应不同局部模式。

不使用膨胀卷积时，一维输出长度为

$$L_{out}=\left\lfloor\frac{L_{in}+2P-F}{S}\right\rfloor+1.$$

$F$ 为核大小，$P$ 为两侧填充，$S>0$ 为步长。二维分别对高宽计算；必须保证核能在填充后输入上至少放置一次。输入 $C_{in}$ 通道，$C_{out}$ 个大小 $F_h\times F_w$ 的核，含每通道偏置时参数数为 $C_{out}(F_hF_wC_{in}+1)$，不随图像空间大小直接增长。

自编例：$32\times32\times3$ 输入，6 个 $5\times5$ 核，步长 1、不填充，输出 $28\times28\times6$，参数 $6(25\times3+1)=456$。最大池化通常没有可学习参数，但会丢失精确位置；局部共享带来平移等变性，并不自动保证任意平移下分类结果完全不变。

## 10.4 RNN：共享参数的时间展开 {#ml-10-04}

序列中当前预测可能依赖过去。简单循环网络写为

$$\mathbf h_t=\phi(W_h\mathbf h_{t-1}+W_x\mathbf x_t+\mathbf b),\qquad\mathbf o_t=W_o\mathbf h_t+\mathbf c.$$

同一个 $W_h,W_x$ 在各时刻复用。展开后类似很深的前馈计算图，训练损失 $J=\sum_t\ell_t$。令 $\mathbf u_t$ 为隐层激活前值、$\boldsymbol\delta_t=\partial J/\partial\mathbf u_t$，若输出在时刻 $t$ 的梯度为 $\mathbf e_t=\partial\ell_t/\partial\mathbf o_t$，则

$$\boldsymbol\delta_t=(W_o^T\mathbf e_t+W_h^T\boldsymbol\delta_{t+1})\odot\phi'(\mathbf u_t),\qquad\boldsymbol\delta_{T+1}=0.$$

于是 $\partial J/\partial W_h=\sum_t\boldsymbol\delta_t\mathbf h_{t-1}^T$。必须把共享参数在各时间位置的贡献相加，不能只更新最后一步。截断 BPTT 只回传有限时间窗，减少计算但改变了梯度估计；它不是完整 BPTT 的精确结果。课件 S18 21 页明确列有“截断深度为 h 的学习（课堂作业）”，本版保存其来源，原页图中细节尚待完整视觉转录。

## 10.5 注意力、Transformer 与预训练 {#ml-10-05}

固定长度状态难以保留长输入全部细节，注意力允许当前位置按相关性读取其他位置。标准缩放点积形式为

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V.$$

$Q\in\mathbb R^{n_q\times d_k}$ 是查询，$K\in\mathbb R^{n_k\times d_k}$ 是键，$V\in\mathbb R^{n_k\times d_v}$ 是值；softmax 沿键位置归一化。掩码 $M$ 在允许位置为 0、禁止位置为负无穷，因果掩码防止自回归训练偷看未来。实现时禁止整行全部被掩盖，以免归一化无定义。

多头注意力让多个投影子空间分别建立关系；位置编码补充顺序信息，残差与归一化帮助信息和梯度传播。课件介绍 ELMo、BERT、GPT、编码器—解码器与 RLHF；本书只保留稳定的机制说明，不把某个版本的参数量或排名写成通用事实。

掩码语言模型根据上下文恢复被遮住的符号；自回归语言模型分解 $p(x_{1:T})=\prod_tp(x_t\mid x_{<t})$。预训练目标、微调目标与下游评估应分别说明。人工反馈对齐通常包含监督微调、偏好或奖励建模、策略优化等环节，不能简化为“模型知道了人的真实意图”。

## 10.6 生成模型与安全边界 {#ml-10-06}

GAN 同时训练生成器 $G$ 与鉴别器 $D$，经典目标为 $\min_G\max_D E_{x\sim p_{data}}\log D(x)+E_{z\sim p(z)}\log[1-D(G(z))]$。交替优化是一个博弈过程，可能不稳定、出现模式坍塌，不能保证有限训练必然达到理想均衡。

扩散模型的前向过程逐步加噪。常用闭式写法 $\mathbf x_t=\sqrt{\bar\alpha_t}\mathbf x_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$，$\boldsymbol\epsilon\sim N(0,I)$，$\bar\alpha_t$ 为逐步保真系数之积。网络学习给定时间和带噪样本时预测噪声；采样按所选反向过程逐步产生数据。生成的是学习分布中的新样本，不是保证还原某张特定训练图；也不能忽略时间相关系数而反复直接减同一噪声。

课件末尾讨论对抗样本：输入在给定扰动预算内变化，可能导致错误预测。应区分白盒、基于分数、基于决策与迁移等威胁模型，并报告扰动范数、预算和评价条件。这里是课程概念整理，不包含对外部系统的攻击实验。

自组织映射、RBF 网络、RBM、强化学习网络等在 S16 37–58 页概述性出现，本版记录其定位，但不宣称已完整推导这些扩展算法。S19 中应用图片与早期平台列表保留在来源材料中，尚未逐项写成经过核实的现状说明。
