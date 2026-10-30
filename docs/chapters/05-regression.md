# 第五章 回归：从最小二乘到正则化

> 往年资料：S10 4–16 页（问题）、18–31 页（LS/ML）、33–37 页（MAP）、39–48 页（基函数与误差）。推导采用行样本约定，例子为自编。

> **编者注（与课件区别的阅读提示）：** 本章在保留上述往年资料来源框架的基础上，对每条主线公式都补写了“过程讲解”与一维／二维小数值例子；凡属编者额外补充的推导或提示，均标以「编者注」。来源页段只作定位，未逐页核对视觉格式，请勿据此推定考试范围。

## 5.1 回归预测的是数值 {#ml-05-01}

**直觉：** 分类题的答案可以随意改名（猫＝0、狗＝1 或猫＝5、狗＝100 都不影响决策）；回归题的答案是一个具体数值，数值之间的“距离”本身就是信息。房价 200 万与 300 万相差 100 万，这个差距会被损失函数如实计入。

### 5.1.1 符号与数据组织（先定维度，再写公式）

本书采用**行样本**约定（与 `docs/guide.md` 统一符号一致）：

- $N$：样本个数，$N\ge 1$；
- $p$：设计矩阵的列数。若原始特征维度为 $d$，加入一列全 1 当作截距后，$p=d+1$。统一符号表里的 $X\in\mathbb R^{N\times d}$ 指“未增广”的设计矩阵，本章的 $p$ 是其增广后的列数；
- 设计矩阵 $X\in\mathbb R^{N\times p}$，**每一行是一个样本的特征向量**，第 $i$ 行记为 $\mathbf x_i^\top$（把该行视为列向量时记作 $\mathbf x_i\in\mathbb R^p$）；
- 待估参数 $\boldsymbol\beta\in\mathbb R^p$，包含各特征权重与截距；
- 观测响应 $\mathbf y\in\mathbb R^N$，第 $i$ 个分量 $y_i\in\mathbb R$；
- 预测 $\widehat{\mathbf y}=X\boldsymbol\beta\in\mathbb R^N$，第 $i$ 个分量 $\hat y_i=\mathbf x_i^\top\boldsymbol\beta$。

> **记住：** 先写清“谁在第几行、谁在第几列、向量的维度是多少”，再代入公式。行样本约定下，$\hat y_i=\mathbf x_i^\top\boldsymbol\beta$（行向量转置成列向量再点乘），而不是 $\boldsymbol\beta^\top\mathbf x_i$ 之外的其他形式——两者数值相等，但维度顺序必须心里有数。

### 5.1.2 评价准则：MSE、RMSE、MAE

平方误差（squared error）是最常用的回归损失：

$$
\operatorname{MSE}=\frac1N\sum_{i=1}^N(y_i-\hat y_i)^2=\frac1N\|\mathbf y-\widehat{\mathbf y}\|^2,\qquad
\widehat{\mathbf y}=X\boldsymbol\beta.
$$

- $\operatorname{RMSE}=\sqrt{\operatorname{MSE}}$，单位与 $y$ 一致（如“万元”）；
- $\operatorname{MAE}=\frac1N\sum_{i=1}^N|y_i-\hat y_i|$，对个别大残差惩罚更温和。

**小数值例子（体会 MSE 对误差的“平方放大”）：** 设有两个模型在同一测试点上的误差序列：
- 模型 A 残差：$(1,1,1)$，则 $\operatorname{MSE}_A=(1+1+1)/3=1$；
- 模型 B 残差：$(0,0,3)$，则 $\operatorname{MSE}_B=(0+0+9)/3=3$。

两者平均绝对误差都是 $1$，但 MSE 相差 3 倍——因为 MSE 对第 3 个点的残差 $3$ 平方成了 $9$。这说明 **MSE 会严厉惩罚少数“离谱”的预测**，这是它和 MAE 的关键区别。

> **常见误区：** 不同量纲任务的 MSE 不能直接比大小。用“万元”计的房价 MSE 天然比用“厘米”计的人体身高 MSE 大得多，这不是模型好坏，是单位不同。比较前应先标准化或统一量纲。

## 5.2 正规方程为什么成立 {#ml-05-02}

**直觉：** 我们要找一条直线（更一般地说，一个超平面）$X\boldsymbol\beta$，让它“尽量靠近”观测到的 $\mathbf y$。在欧氏距离意义下，“最近的那个列空间里的向量”恰是 $\mathbf y$ 到列空间的正交投影。正规方程就是把“残差垂直于列空间”这条几何事实翻译成代数。

### 5.2.1 从目标函数出发：设定与展开

**假设与维度**
- $X\in\mathbb R^{N\times p}$（行样本），$\boldsymbol\beta\in\mathbb R^p$，$\mathbf y\in\mathbb R^N$；
- 目标：最小化残差平方和 $J(\boldsymbol\beta)=\|\mathbf y-X\boldsymbol\beta\|^2$。

**起始式（写成内积）**
$$
J(\boldsymbol\beta)=\|\mathbf y-X\boldsymbol\beta\|^2=(\mathbf y-X\boldsymbol\beta)^\top(\mathbf y-X\boldsymbol\beta).
$$

**逐步展开**

第 1 步：用分配律把括号拆开。
$$
(\mathbf y-X\boldsymbol\beta)^\top(\mathbf y-X\boldsymbol\beta)
=(\mathbf y^\top-\boldsymbol\beta^\top X^\top)(\mathbf y-X\boldsymbol\beta).
$$

第 2 步：逐项相乘（四个项）。
$$
=\mathbf y^\top\mathbf y-\mathbf y^\top X\boldsymbol\beta-\boldsymbol\beta^\top X^\top\mathbf y+\boldsymbol\beta^\top X^\top X\boldsymbol\beta.
$$

第 3 步：合并两个标量交叉项。注意 $\mathbf y^\top X\boldsymbol\beta$ 是一个标量，标量等于自己的转置，故
$$
\mathbf y^\top X\boldsymbol\beta=(\mathbf y^\top X\boldsymbol\beta)^\top=\boldsymbol\beta^\top X^\top\mathbf y.
$$
两项相加为 $-2\boldsymbol\beta^\top X^\top\mathbf y$。于是
$$
J(\boldsymbol\beta)=\mathbf y^\top\mathbf y-2\boldsymbol\beta^\top X^\top\mathbf y+\boldsymbol\beta^\top X^\top X\boldsymbol\beta.
$$

**求导（对向量 $\boldsymbol\beta$ 的梯度）**

用两条向量微分规则（对列向量 $\boldsymbol\beta$）：
- $\nabla_{\boldsymbol\beta}(\mathbf a^\top\boldsymbol\beta)=\mathbf a$；
- $\nabla_{\boldsymbol\beta}(\boldsymbol\beta^\top M\boldsymbol\beta)=2M\boldsymbol\beta$，当 $M$ 对称（这里 $M=X^\top X$ 显然对称）。

$$
\nabla_{\boldsymbol\beta} J=-2X^\top\mathbf y+2X^\top X\boldsymbol\beta.
$$

**令梯度为零，得到正规方程**
$$
-2X^\top\mathbf y+2X^\top X\boldsymbol\beta=0
\quad\Longrightarrow\quad
X^\top X\boldsymbol\beta=X^\top\mathbf y.
\tag{5.1}
$$

这就是**正规方程（normal equation）**。

> **编者注（为什么是“最小”而不是“驻点”就行）：** $J(\boldsymbol\beta)$ 是 $\boldsymbol\beta$ 的二次型且二次型矩阵 $X^\top X$ 半正定，因此它是凸函数；梯度为零的驻点就是全局最小点。无需再检验二阶充分条件。

### 5.2.2 解的唯一性、满列秩与正定性

**假设：** $X$ **满列秩**，即 $\operatorname{rank}(X)=p$（要求 $N\ge p$）。

**证明 $X^\top X$ 正定：**
任取非零向量 $\mathbf v\in\mathbb R^p$，
$$
\mathbf v^\top(X^\top X)\mathbf v=(X\mathbf v)^\top(X\mathbf v)=\|X\mathbf v\|^2.
$$
因为 $X$ 满列秩，$X\mathbf v\neq\mathbf 0$（否则 $\mathbf v$ 落在零空间），故 $\|X\mathbf v\|^2>0$。所以 $X^\top X$ 正定，从而可逆。

**唯一解**
$$
\widehat{\boldsymbol\beta}=(X^\top X)^{-1}X^\top\mathbf y.
\tag{5.2}
$$

**结论与成立条件：** 在“行样本、满列秩、欧氏平方损失”三条件下，最小二乘解唯一，且由式 (5.2) 给出。

### 5.2.3 几何解释：正交投影（最直观的“为什么成立”）

把正规方程 (5.1) 移项：
$$
X^\top(\mathbf y-X\widehat{\boldsymbol\beta})=0
\quad\Longrightarrow\quad
X^\top\mathbf r=0,\ \mathbf r=\mathbf y-X\widehat{\boldsymbol\beta}.
$$

$X^\top\mathbf r$ 的第 $j$ 个分量正是“设计矩阵第 $j$ 列”与残差 $\mathbf r$ 的内积，因此 $X^\top\mathbf r=0$ 表示：**残差 $\mathbf r$ 与设计矩阵的每一列都正交**，即 $\mathbf r\perp\mathcal C(X)$（$\mathcal C(X)$ 为 $X$ 的列空间）。

于是 $X\widehat{\boldsymbol\beta}$ 是 $\mathbf y$ 在 $\mathcal C(X)$ 上的**正交投影**，投影矩阵为
$$
P=X(X^\top X)^{-1}X^\top,\qquad \widehat{\mathbf y}=P\mathbf y.
$$
容易验证 $P$ 是幂等且对称的（$P^2=P,\ P^\top=P$），这是投影算子的标准性质。

**二维小数值例子（让投影“看得见”）：** 取 5.2.5 的例，
$X=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix}$，$\mathbf y=(1,2,2)^\top$，由后面算得
$\widehat{\boldsymbol\beta}=(7/6,\,1/2)^\top$，于是投影
$$
\widehat{\mathbf y}=X\widehat{\boldsymbol\beta}=\left(\tfrac76,\ \tfrac53,\ \tfrac{13}{6}\right)^\top,
\quad
\mathbf r=\mathbf y-\widehat{\mathbf y}=\left(-\tfrac16,\ \tfrac13,\ -\tfrac16\right)^\top.
$$
取 $X$ 的两列 $\mathbf c_1=(1,1,1)^\top,\ \mathbf c_2=(0,1,2)^\top$，检查正交性：
$$
\mathbf r^\top\mathbf c_1=-\tfrac16+\tfrac13-\tfrac16=0,\qquad
\mathbf r^\top\mathbf c_2=0\cdot(-\tfrac16)+1\cdot\tfrac13+2\cdot(-\tfrac16)=\tfrac13-\tfrac13=0.
$$
残差同时垂直于两列，几何上正是“投影残差垂直于投影平面”，与代数结论一致。

### 5.2.4 列不满秩时：参数不唯一、伪逆取最小范数解

**假设放宽：** $X$ 列相关，即 $\operatorname{rank}(X)=r<p$。

此时存在非零 $\mathbf v$ 使 $X\mathbf v=\mathbf 0$，于是
$$
\mathbf v^\top X^\top X\mathbf v=\|X\mathbf v\|^2=0,
$$
故 $X^\top X$ **半正定但不正定**，不可逆，正规方程 (5.1) 有无穷多解。

**关键事实：** 虽然 $\boldsymbol\beta$ 不唯一，但“投影” $X\widehat{\boldsymbol\beta}$ 唯一。因为任意两个解相差一个零空间向量 $\mathbf v$（即 $\widehat{\boldsymbol\beta}_2=\widehat{\boldsymbol\beta}_1+\mathbf v$ 仍满足方程），而
$$
X\widehat{\boldsymbol\beta}_2=X\widehat{\boldsymbol\beta}_1+X\mathbf v=X\widehat{\boldsymbol\beta}_1,
$$
投影不变。

**最小范数解（Moore–Penrose 伪逆）：**
$$
\widehat{\boldsymbol\beta}^+=X^+\mathbf y,\qquad
X^+=(X^\top X)^+X^\top,
$$
其中 $(\cdot)^+$ 表示伪逆。在“所有令 $X\boldsymbol\beta$ 等于同一投影的解”中，$\boldsymbol\beta^+$ 是欧氏范数 $\|\boldsymbol\beta\|$ 最小的那一个。

> **记住：** 列相关 ≠ 模型不能预测。预测（投影）仍然唯一；不唯一的是“参数本身”。伪逆在 $X^\top X$ 奇异时给出稳定、最小范数的 $\boldsymbol\beta$。

**数值计算建议：** 不要直接算 $(X^\top X)^{-1}$（先平方条件数、再求逆会放大误差）。优先用 QR 分解、SVD，或直接求解线性方程组 $X^\top X\boldsymbol\beta=X^\top\mathbf y$。

### 5.2.5 自编例题（核对数字，并用正规方程复算）

**数据** $(x,y)=(0,1),(1,2),(2,2)$，模型 $y=ax+b$。

**矩阵写法（行样本）：** 取每行 $(1,\ x_i)$，则
$$
X=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix},\quad
\boldsymbol\beta=\begin{bmatrix}b\\ a\end{bmatrix},\quad
\mathbf y=\begin{bmatrix}1\\2\\2\end{bmatrix}.
$$

**算 $X^\top X$ 与 $X^\top\mathbf y$：**
$$
X^\top X=
\begin{bmatrix}1&1&1\\0&1&2\end{bmatrix}
\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix}
=\begin{bmatrix}3&3\\3&5\end{bmatrix},
\qquad
X^\top\mathbf y=
\begin{bmatrix}1&1&1\\0&1&2\end{bmatrix}
\begin{bmatrix}1\\2\\2\end{bmatrix}
=\begin{bmatrix}5\\6\end{bmatrix}.
$$

**正规方程：**
$$
\begin{bmatrix}3&3\\3&5\end{bmatrix}\begin{bmatrix}b\\a\end{bmatrix}
=\begin{bmatrix}5\\6\end{bmatrix}.
$$
由第一行 $3b+3a=5$ 得 $b+a=5/3$；代入第二行 $3b+5a=6$ 消去 $b$：
$$
3(\tfrac53-a)+5a=6\ \Rightarrow\ 5-3a+5a=6\ \Rightarrow\ 2a=1\ \Rightarrow\ a=\tfrac12.
$$
于是 $b=\tfrac53-\tfrac12=\tfrac{10-3}{6}=\tfrac76$。**与课件公式结果一致**：$a=1/2,\ b=7/6$。

**残差与 MSE（逐一核算）：**
- $x=0$：$\hat y=7/6$，残差 $1-7/6=-1/6$；
- $x=1$：$\hat y=1/2+7/6=5/3$，残差 $2-5/3=1/3$；
- $x=2$：$\hat y=1+7/6=13/6$，残差 $2-13/6=-1/6$。

残差平方和：
$$
(-\tfrac16)^2+(\tfrac13)^2+(-\tfrac16)^2=\tfrac1{36}+\tfrac4{36}+\tfrac1{36}=\tfrac6{36}=\tfrac16.
$$
$$
\operatorname{MSE}=\frac1N\cdot\tfrac16=\tfrac{1}{18}.
$$

> **编者注（与一元公式对照）：** 课件用中心化公式
> $a=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}$、$b=\bar y-a\bar x$，
> 其中 $\bar x=1,\ \bar y=5/3$。分子 $=(-1)(-2/3)+0(1/3)+(1)(1/3)=1$，分母 $=2$，得 $a=1/2$，$b=5/3-1/2=7/6$，与正规方程结果完全相同——因为一元公式本身就是正规方程在 $p=2$ 时的特例。

## 5.3 高斯误差连接最小二乘与 ML {#ml-05-03}

**直觉：** 最小二乘“不问误差来源”，只看距离最小；极大似然“问误差分布”。当误差恰好是高斯（正态）时，两者殊途同归：让“出现这组数据的概率最大”等价于“让残差平方和最小”。

### 5.3.1 设定：高斯噪声模型

**假设**
- $\mathbf y=X\boldsymbol\beta+\boldsymbol\epsilon$，其中 $\boldsymbol\epsilon=(\epsilon_1,\dots,\epsilon_N)^\top$；
- 各误差独立同分布，$\epsilon_i\sim N(0,\sigma^2)$，$\sigma^2>0$ 已知或待估；
- 给定 $X$ 与 $\boldsymbol\beta$，则 $y_i\mid \mathbf x_i\sim N(\mathbf x_i^\top\boldsymbol\beta,\ \sigma^2)$，整体
  $\mathbf y\mid X\sim N(X\boldsymbol\beta,\ \sigma^2 I_N)$。

### 5.3.2 似然、对数似然与负对数似然（逐步推导）

**写出似然（联合密度）**
$$
p(\mathbf y\mid X,\boldsymbol\beta,\sigma^2)
=\prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y_i-\mathbf x_i^\top\boldsymbol\beta)^2}{2\sigma^2}\right)
=(2\pi\sigma^2)^{-N/2}\exp\!\left(-\frac{\|\mathbf y-X\boldsymbol\beta\|^2}{2\sigma^2}\right).
$$

**取对数似然**
$$
\ell(\boldsymbol\beta)
=\log p(\mathbf y\mid X,\boldsymbol\beta,\sigma^2)
=-\frac N2\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\|\mathbf y-X\boldsymbol\beta\|^2.
$$

**负对数似然（要最小化的目标）**
$$
-\ell(\boldsymbol\beta)=\frac N2\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}\|\mathbf y-X\boldsymbol\beta\|^2.
\tag{5.3}
$$

### 5.3.3 为什么 MLE = 最小二乘

对 $\boldsymbol\beta$ 最小化 (5.3)：
- 第一项 $\frac N2\log(2\pi\sigma^2)$ 与 $\boldsymbol\beta$ 无关，可丢弃；
- 第二项系数 $\frac{1}{2\sigma^2}>0$ 是常数，不影响最优点位置。

因此“最小化负对数似然”等价于“最小化 $\|\mathbf y-X\boldsymbol\beta\|^2$”，即最小二乘。最优参数相同：
$$
\widehat{\boldsymbol\beta}_{\mathrm{ML}}=\widehat{\boldsymbol\beta}_{\mathrm{LS}}=(X^\top X)^{-1}X^\top\mathbf y\quad(\text{满列秩时}).
$$

**结论与成立条件：** 在“误差独立同分布、零均值、同方差、服从高斯”前提下，MLE 与 LS 给出相同估计。**“等价”指目标函数等价，并不表示现实误差都该服从高斯**；若误差非高斯，LS 与 ML 一般会分道扬镳。

### 5.3.4 梯度下降视角：批量梯度与 SGD

把 LS 目标写成**平均损失**（便于跨数据量比较）：
$$
L(\boldsymbol\beta)=\frac{1}{2N}\|X\boldsymbol\beta-\mathbf y\|^2.
$$

**全样本（批量）梯度：** 对 (5.3) 的 $\|X\boldsymbol\beta-\mathbf y\|^2/(2N)$ 求梯度，
$$
\nabla_{\boldsymbol\beta} L=\frac1N X^\top(X\boldsymbol\beta-\mathbf y).
\tag{5.4}
$$
批量梯度下降更新：$\boldsymbol\beta\leftarrow\boldsymbol\beta-\eta\,\nabla_{\boldsymbol\beta}L$。

**随机梯度下降（SGD）：** 每次取一个样本 $i$，用其梯度近似整体梯度。第 $i$ 个样本对平均损失 $L$ 的贡献为 $\frac{1}{2N}( \mathbf x_i^\top\boldsymbol\beta-y_i)^2$，其梯度为
$$
\nabla_{\boldsymbol\beta}^{(i)}L=\frac1N(\mathbf x_i^\top\boldsymbol\beta-y_i)\mathbf x_i.
\tag{5.5}
$$
若不除以 $N$（即优化未平均的总损失 $\frac12(\mathbf x_i^\top\boldsymbol\beta-y_i)^2$），则梯度为 $(\mathbf x_i^\top\boldsymbol\beta-y_i)\mathbf x_i$——这正是课件所写形式。**两种写法只差常数倍 $1/N$，本质相同，但比较学习率时要先分清用的是哪种目标。**

> **常见误区：** 学习率 $\eta$ 过大时，参数在最优解两侧来回跳跃甚至发散。直觉上，若 $X^\top X$ 的最大特征值很大，一步跨得太远就会越过谷底。对特征做标准化/缩放可改善条件数，使梯度下降更平稳。

## 5.4 MAP 与岭回归 {#ml-05-04}

**直觉：** 最小二乘“惜墨如金”地拟合数据，数据少或特征相关时容易过拟合、系数乱跳。岭回归给参数加一个“偏好：系数别太大”的先验（高斯先验），于是把 LS 目标加上 $\lambda\|\boldsymbol\beta\|^2$ 的惩罚——这等价于贝叶斯里的 MAP（最大后验）。LASSO 换成 $L_1$ 惩罚，能把不重要系数压成精确的 0。

### 5.4.1 MAP 设定：参数的高斯先验

**假设**
- 似然同 5.3：$\mathbf y\mid X,\boldsymbol\beta\sim N(X\boldsymbol\beta,\sigma^2 I)$；
- 参数先验 $\boldsymbol\beta\sim N(\mathbf 0,\tau^2 I_p)$，$\tau^2>0$，即各分量独立、零均值、方差 $\tau^2$。

**后验（贝叶斯公式）：**
$$
p(\boldsymbol\beta\mid \mathbf y,X)\propto p(\mathbf y\mid X,\boldsymbol\beta)\,p(\boldsymbol\beta).
$$
MAP 即最大化后验，等价于最小化**负对数后验**。

### 5.4.2 负对数后验的逐步推导

由 5.3 已知负对数似然：
$$
-\log p(\mathbf y\mid X,\boldsymbol\beta)=\frac N2\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}\|\mathbf y-X\boldsymbol\beta\|^2.
$$

**写负对数先验：**
$$
-\log p(\boldsymbol\beta)=\frac p2\log(2\pi\tau^2)+\frac{1}{2\tau^2}\|\boldsymbol\beta\|^2.
$$

**负对数后验 = 二者相加（再加与 $\boldsymbol\beta$ 无关的常数）：**
$$
-\log p(\boldsymbol\beta\mid \mathbf y,X)
=\frac N2\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}\|\mathbf y-X\boldsymbol\beta\|^2
+\frac p2\log(2\pi\tau^2)+\frac{1}{2\tau^2}\|\boldsymbol\beta\|^2+\text{const}.
$$

**只保留含 $\boldsymbol\beta$ 的项并同乘 $2\sigma^2$：**
$$
\|\mathbf y-X\boldsymbol\beta\|^2+\frac{\sigma^2}{\tau^2}\|\boldsymbol\beta\|^2.
$$
记
$$
\lambda=\frac{\sigma^2}{\tau^2}>0,
\tag{5.6}
$$
得到**岭回归目标**：
$$
J_{\text{ridge}}(\boldsymbol\beta)=\|\mathbf y-X\boldsymbol\beta\|^2+\lambda\|\boldsymbol\beta\|^2.
\tag{5.7}
$$

### 5.4.3 岭回归的正规方程

对 (5.7) 求梯度（沿用 5.2 的微分规则，并加 $\nabla(\lambda\|\boldsymbol\beta\|^2)=2\lambda\boldsymbol\beta$）：
$$
\nabla_{\boldsymbol\beta} J_{\text{ridge}}=-2X^\top(\mathbf y-X\boldsymbol\beta)+2\lambda\boldsymbol\beta.
$$
令为零：
$$
-2X^\top\mathbf y+2X^\top X\boldsymbol\beta+2\lambda\boldsymbol\beta=0
\ \Rightarrow\
(X^\top X+\lambda I)\boldsymbol\beta=X^\top\mathbf y.
\tag{5.8}
$$

**解（注意 $\lambda>0$ 保证可逆）：**
$$
\widehat{\boldsymbol\beta}_{\text{ridge}}=(X^\top X+\lambda I)^{-1}X^\top\mathbf y.
\tag{5.9}
$$

**正定性论证（为何岭回归不怕列相关）：** 任取 $\mathbf v\neq\mathbf 0$，
$$
\mathbf v^\top(X^\top X+\lambda I)\mathbf v=\|X\mathbf v\|^2+\lambda\|\mathbf v\|^2>0,
$$
因为 $\|X\mathbf v\|^2\ge0$ 且 $\lambda\|\mathbf v\|^2>0$。所以 $X^\top X+\lambda I$ 必正定、必可逆——**即使 $X^\top X$ 奇异，岭回归仍有唯一解**。

**收缩效应（沿特征向量方向看）：** 设 $X^\top X$ 的单位特征向量为 $\mathbf u$、特征值 $\gamma\ge0$。在 $\mathbf u$ 方向上，岭系数相对 LS 系数被乘了因子 $\frac{\gamma}{\gamma+\lambda}$。$\gamma$ 越小（该方向越不稳定）收缩越强；$\lambda\to0$ 回到 LS，$\lambda\to\infty$ 系数趋零。

### 5.4.4 截距不惩罚 与 $\lambda$ 的 $N$ 换算（两处易错点）

**截距（偏置）通常不应被惩罚。** 若把截距也压向 0，模型会被迫用截距去“抵消”惩罚，扭曲斜率。标准做法是只对权重加惩罚：把 (5.8) 中的 $I$ 换成“截距位为 0、其余为 1”的对角矩阵 $D$，即
$$
(X^\top X+\lambda D)\boldsymbol\beta=X^\top\mathbf y,\quad D=\operatorname{diag}(0,1,\dots,1).
$$
这对应“先验只加在权重上、截距用无信息先验”。

> **常见误区：** 不同教材对岭目标用“总损失” $\|.\|^2+\lambda\|\boldsymbol\beta\|^2$ 还是“平均损失” $\frac1N\|.\|^2+\lambda\|\boldsymbol\beta\|^2$ 定义不同。若用平均损失，式 (5.6) 变为 $\lambda=\sigma^2/(N\tau^2)$，多一个 $N$。**跨资料比较 $\lambda$ 数值前，必须先确认对方用的是哪种损失定义。**

### 5.4.5 岭回归小数值例子（看“收缩”与“训练误差交换”）

仍用 5.2.5 的数据 $(0,1),(1,2),(2,2)$，模型 $y=ax+b$，**只惩罚斜率 $a$**（截距不罚），即目标 $\|.\|^2+\lambda a^2$。

先中心化：令 $\tilde x_i=x_i-\bar x$，$\tilde y_i=y_i-\bar y$，$\bar x=1,\ \bar y=5/3$。则截距最优为 $b=\bar y-a\bar x=5/3-a$（截距不罚时其估计公式不变）。残差平方和只在中心化量上依赖 $a$：
$$
S(a)=\sum_i\bigl(\tilde y_i-a\tilde x_i\bigr)^2
=\underbrace{\sum\tilde y_i^2}_{2/3}-2a\underbrace{\sum\tilde x_i\tilde y_i}_{1}+a^2\underbrace{\sum\tilde x_i^2}_{2}.
$$
加惩罚：
$$
J(a)=\tfrac23-2a+2a^2+\lambda a^2=\tfrac23-2a+(2+\lambda)a^2.
$$
$$
\frac{dJ}{da}=-2+2(2+\lambda)a=0\ \Rightarrow\ a=\frac{1}{2+\lambda}.
$$
- $\lambda=0$：$a=1/2$（回到 LS）；
- $\lambda=1$：$a=1/3$，斜率被“收缩”向 0；
- $\lambda\to\infty$：$a\to0$，模型退化为常数。

取 $\lambda=1$ 算训练 MSE（验证“用训练误差换稳定性”）：
$b=5/3-1/3=4/3$，预测 $(4/3,\ 5/3,\ 2)$，残差 $(-1/3,\ 1/3,\ 0)$，平方和 $2/9$，$\operatorname{MSE}=2/27\approx0.074$，比 LS 的 $1/18\approx0.056$ 略大。**这正是预期**：在这 3 个点上岭回归牺牲了一点训练拟合，换取更小的系数与更稳的外推——偏差—方差视角（5.5）会解释它为何可能降低总体预测误差。

### 5.4.6 LASSO 与 $L_1$ 惩罚：精确零系数

LASSO 把岭回归的 $L_2$ 惩罚换成 $L_1$：
$$
J_{\text{lasso}}(\boldsymbol\beta)=\|\mathbf y-X\boldsymbol\beta\|^2+\lambda\sum_{j=1}^p|\beta_j|.
\tag{5.10}
$$

**为什么能给出精确零？** $L_2$ 球是光滑椭圆，最优解一般不会恰好落在坐标轴上；而 $L_1$ 球是带尖角（角落在坐标轴上）的菱形。最小化凸目标时，解容易“卡”在尖角上，对应某些 $\beta_j=0$——从而实现**自动特征选择**。岭回归通常只收缩、不产生精确零，不能默认当作离散特征选择工具。

> **记住：** $L_1$→稀疏（精确零、可做选择）；$L_2$→收缩（趋零但不为零、稳方向）。这是两者最实用的区别。

## 5.5 非线性输入与偏差—方差 {#ml-05-05}

**直觉：** 数据可能不是直线，但我们不必放弃“对参数线性”这一强大工具——把 $x$ 换成一组基函数 $\phi(x)$（比如多项式、正弦），模型对输入非线性、对参数仍线性，最小二乘照用。代价是：基函数越复杂越容易过拟合。偏差—方差分解则告诉我们“过拟合”到底损失在哪：预测误差＝偏差²＋方差＋噪声。

### 5.5.1 基函数：对输入非线性，对参数仍线性

令
$$
\boldsymbol\phi(x)=\bigl(\phi_1(x),\dots,\phi_p(x)\bigr)^\top,
$$
例如多项式基 $\boldsymbol\phi(x)=(1,\ x,\ x^2,\dots,x^m)^\top$。模型为
$$
f(x)=\boldsymbol\beta^\top\boldsymbol\phi(x)=\sum_{j=1}^p\beta_j\phi_j(x).
$$
**关键点：** 虽然 $f$ 对 $x$ 是多项式（非线性），但对 $\boldsymbol\beta$ 是线性的——只要把“特征”从 $x$ 换成 $\boldsymbol\phi(x)$，5.2 的全部最小二乘结论直接适用，只是把设计矩阵的每行换成 $\boldsymbol\phi(\mathbf x_i)^\top$。

> **常见误区：** “训练误差更小”不保证外推更可靠。高阶多项式可能在 $[0,2]$ 内拟合得极好，却在 $x=10$ 处剧烈震荡（龙格现象式行为）。基函数带来表达力，也带来过拟合风险，需要靠正则化或交叉验证来驾驭。

### 5.5.2 偏差—方差分解的设定

**假设（逐条写清）**
- 真实关系：$Y=f_*(x)+\epsilon$，其中 $f_*(x)$ 是未知真函数；
- 噪声零均值、条件方差有限：$\mathbb E[\epsilon\mid x]=0$，$\operatorname{Var}(\epsilon\mid x)=\sigma^2(x)$；
- 训练集 $D$ 是随机抽取的（含 $N$ 个样本），其随机性记为下标 $D$；
- 用 $D$ 训练得到模型 $\hat f_D(x)$（$\hat f$ 依赖 $D$，故下标 $D$）；
- 在**固定输入 $x$** 处考察预测误差，对训练集随机性与噪声同时取期望：
  $$
  \operatorname{Err}(x)=\mathbb E_{D,\epsilon}\big[(Y-\hat f_D(x))^2\big].
$$

### 5.5.3 逐步推导：加减 $\bar f$ 并利用零均值消去交叉项

**第 1 步：定义平均预测**
$$
\bar f(x)=\mathbb E_D[\hat f_D(x)]
$$
（把不同训练集训出的模型在 $x$ 处的预测取平均）。

**第 2 步：在误差里加、减 $\bar f(x)$**
$$
Y-\hat f_D(x)
=\bigl(Y-f_*(x)\bigr)+\bigl(f_*(x)-\bar f(x)\bigr)+\bigl(\bar f(x)-\hat f_D(x)\bigr).
$$
记三段分别为 $A,\ B,\ C$：
$$
A=Y-f_*(x),\quad B=f_*(x)-\bar f(x),\quad C=\bar f(x)-\hat f_D(x).
$$
注意：$A$ 含噪声 $\epsilon$；$B$ 只依赖 $x$、与训练集 $D$ 及噪声 $\epsilon$ 都无关（因 $f_*$、$\bar f$ 不含随机性）；$C$ 依赖训练集 $D$ 但不含 $\epsilon$。

**第 3 步：展开平方**
$$
\mathbb E[(A+B+C)^2]
=\mathbb E[A^2]+\mathbb E[B^2]+\mathbb E[C^2]
+2\mathbb E[AB]+2\mathbb E[AC]+2\mathbb E[BC].
$$

**第 4 步：逐项处理，重点证明三个交叉项为零**

- $\mathbb E[A^2]$：
  $$
  \mathbb E_\epsilon[A^2]=\mathbb E_\epsilon[(Y-f_*(x))^2]=\mathbb E[\epsilon^2]=\sigma^2(x)
  \quad(\text{因 }\mathbb E[\epsilon]=0).
  $$

- $\mathbb E[B^2]$：$B$ 与随机性无关，故 $\mathbb E[B^2]=B^2=\bigl(f_*(x)-\bar f(x)\bigr)^2$。这正是**偏差平方**。

- $\mathbb E[C^2]$：$C=\bar f(x)-\hat f_D(x)$，于是
  $$
  \mathbb E_D[C^2]=\mathbb E_D\bigl[(\hat f_D(x)-\bar f(x))^2\bigr]=\operatorname{Var}_D[\hat f_D(x)].
  $$
  这正是**方差**。

- $\mathbb E[AB]=0$：先对 $\epsilon$ 取条件期望，$B$ 视作常数，
  $$
  \mathbb E[AB]=\mathbb E_D\bigl[B\cdot\mathbb E_\epsilon[A\mid D]\bigr]
  =\mathbb E_D\bigl[B\cdot\mathbb E_\epsilon[Y-f_*(x)\mid D]\bigr]
  =\mathbb E_D[B\cdot 0]=0.
  $$

- $\mathbb E[AC]=0$：给定 $D$ 时 $C$ 是常数（不含 $\epsilon$），
  $$
  \mathbb E[AC]=\mathbb E_D\bigl[C\cdot\mathbb E_\epsilon[A\mid D]\bigr]
  =\mathbb E_D[C\cdot 0]=0.
  $$

- $\mathbb E[BC]=0$：这里 $\bar f(x)=\mathbb E_D[\hat f_D(x)]$。展开
  $$
  \mathbb E_D[(f_*-\bar f)(\bar f-\hat f)]
  =\mathbb E_D\bigl[f_*\bar f-f_*\hat f-\bar f^2+\bar f\hat f\bigr]
  =f_*\bar f-f_*\bar f-\bar f^2+\bar f^2=0.
  $$
  （用到了 $\mathbb E_D[\hat f]=\bar f$。）

**第 5 步：合并得三项分解**
$$
\boxed{\;
\mathbb E_{D,Y}\big[(Y-\hat f_D(x))^2\big]
=\bigl[\bar f(x)-f_*(x)\bigr]^2+\operatorname{Var}_D[\hat f_D(x)]+\sigma^2(x)
\;}
\tag{5.11}
$$
即
$$
\text{期望预测误差}=\text{偏差}^2+\text{方差}+\text{不可约噪声}.
$$

**结论与成立条件：** 分解只需要“噪声零均值、方差有限、训练集随机抽取”这几条；它不要求模型是高斯、也不要求线性。偏差衡量“平均预测偏离真函数多远”，方差衡量“不同训练集训出的模型波动多大”，$\sigma^2(x)$ 是再好的模型也去不掉的噪声下限。

### 5.5.4 用偏差—方差理解岭回归与最小二乘

- **最小二乘的无偏性有前提：** 说“LS 是无偏的”需要两层假设——(1) 模型形式正确（真函数确实在所拟合的基函数空间内，即 $f_*\in\operatorname{span}\{\phi_j\}$）；(2) 噪声零均值 $\mathbb E[\epsilon\mid X]=0$。若真函数不在该空间（模型误设），平均预测 $\bar f$ 仍偏离 $f_*$，偏差不为零。无偏不是无需前提的普遍结论。
- **岭回归主动引入偏差以降低方差：** 把系数向零收缩，使 $\operatorname{Var}_D[\hat f]$ 下降；当方差降得比偏差升得多时，总误差 (5.11) 下降。所以“岭回归通常更好”是**期望意义下**的权衡，并不保证对每一条具体数据都改善。

> **记住：** 复杂度↑→偏差↓、方差↑；复杂度↓→偏差↑、方差↓。最优模型在两者间取得平衡，而非“训练误差最低”的模型。

## 5.6 本章小结与练习 {#ml-05-06}

**本章主线回顾**
1. 回归预测实数，用 MSE/RMSE/MAE 评价；MSE 对大残差平方放大，且跨量纲不可直接比较。
2. 正规方程 $X^\top X\boldsymbol\beta=X^\top\mathbf y$ 由最小化 $\|\mathbf y-X\boldsymbol\beta\|^2$ 求导得到；满列秩时解唯一 $\boldsymbol\beta=(X^\top X)^{-1}X^\top\mathbf y$；几何上是 $\mathbf y$ 到列空间的正交投影。
3. 高斯噪声下，负对数似然与 LS 目标只差常数，故 MLE = LS。
4. 高斯先验 $\boldsymbol\beta\sim N(0,\tau^2I)$ 下 MAP 给出岭回归 $(X^\top X+\lambda I)\boldsymbol\beta=X^\top\mathbf y$，$\lambda=\sigma^2/\tau^2$；LASSO 用 $L_1$ 可得精确零系数。
5. 基函数让模型对输入非线性、对参数仍线性；偏差—方差分解 $\mathbb E[(Y-\hat f)^2]=\text{偏差}^2+\text{方差}+\text{噪声}$ 解释了过拟合与正则化的本质。

**练习：** 本章对应习题见 [练习](../exercises.md)。建议先遮住解答，独立写出 5.2 的正规方程推导与 5.5 的偏差—方差三项展开，再对照检查假设、维度与交叉项消去步骤。
