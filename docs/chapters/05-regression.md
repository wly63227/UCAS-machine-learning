# 第五章 回归：从最小二乘到正则化

> 往年资料：S10 4–16 页（问题）、18–31 页（LS/ML）、33–37 页（MAP）、39–48 页（基函数与误差）。推导采用行样本约定，例子为自编。

## 5.1 回归预测的是数值 {#ml-05-01}

分类中的类别编号可以改名，回归中的数值大小不能任意改动。房价预测中的 200 万和 300 万相差 100 万，平方损失会将这个差异计入目标。

对 $N$ 个样本，令增广设计矩阵 $X\in\mathbb R^{N\times p}$，每行是一个样本的特征，若含截距则有一列全 1；$\boldsymbol\beta\in\mathbb R^p$ 包含权重和截距，$\mathbf y\in\mathbb R^N$ 为观测响应。模型是 $\widehat{\mathbf y}=X\boldsymbol\beta$。

常用评价为 $MSE=N^{-1}\sum_i(y_i-\hat y_i)^2$，RMSE 是其平方根，具有响应的单位；MAE 为平均绝对误差，对大残差的惩罚较缓。不同量纲任务的 MSE 不能直接比较。

## 5.2 正规方程为什么成立 {#ml-05-02}

最小二乘使残差平方和 $J=\|\mathbf y-X\boldsymbol\beta\|^2$ 最小。展开为 $\mathbf y^T\mathbf y-2\boldsymbol\beta^TX^T\mathbf y+\boldsymbol\beta^TX^TX\boldsymbol\beta$，求导得到

$$X^TX\widehat{\boldsymbol\beta}=X^T\mathbf y.$$

当 $X$ 满列秩，$X^TX$ 正定，解唯一。几何上残差满足 $X^T(\mathbf y-X\widehat{\boldsymbol\beta})=0$，即残差垂直于所有设计矩阵列，预测是 $\mathbf y$ 到列空间的正交投影。

列相关时参数不唯一，但投影仍唯一；可用伪逆取最小范数解。数值计算优先使用 QR、SVD 或线性方程求解，直接求逆可能放大误差。

自编例：$(x,y)=(0,1),(1,2),(2,2)$，模型 $ax+b$。$\bar x=1,\bar y=5/3$，$a=\sum_i(x_i-\bar x)(y_i-\bar y)/\sum_i(x_i-\bar x)^2=1/2$，$b=7/6$。残差依次为 $-1/6,1/3,-1/6$，平方和 $1/6$，MSE 为 $1/18$。

## 5.3 高斯误差连接最小二乘与 ML {#ml-05-03}

假设 $\mathbf y=X\boldsymbol\beta+\boldsymbol\epsilon$，各误差独立同分布且 $\epsilon_i\sim N(0,\sigma^2)$。固定 $X$ 和 $\sigma^2>0$，负对数似然为

$$-\ell(\boldsymbol\beta)=\frac N2\log(2\pi\sigma^2)+\frac1{2\sigma^2}\|\mathbf y-X\boldsymbol\beta\|^2.$$

第一项与参数无关，第二项只有一个正比例系数，所以 ML 与最小二乘有相同的最优参数。“等价”依赖此噪声假设；并不表示所有回归误差都应服从高斯。

对目标 $\|X\boldsymbol\beta-\mathbf y\|^2/(2N)$，批量梯度为 $X^T(X\boldsymbol\beta-\mathbf y)/N$；SGD 用一个样本的 $(\mathbf x_i^T\boldsymbol\beta-y_i)\mathbf x_i$ 近似。学习率过大可能发散，特征缩放可改善条件数。

## 5.4 MAP 与岭回归 {#ml-05-04}

参数先验为 $\boldsymbol\beta\sim N(0,\tau^2I)$ 时，将负对数先验加到负对数似然，再乘 $2\sigma^2$，得到

$$\|\mathbf y-X\boldsymbol\beta\|^2+\lambda\|\boldsymbol\beta\|^2,\qquad\lambda=\sigma^2/\tau^2.$$

驻点满足 $(X^TX+\lambda I)\widehat{\boldsymbol\beta}=X^T\mathbf y$。$\lambda>0$ 让矩阵正定，减小不稳定方向的权重。若不惩罚截距，应把 $I$ 换成相应位置为零的对角矩阵，并明确先验只加在权重上。使用平均损失时 $\lambda$ 的换算还会含 $N$，不可跨定义直接比较数值。

LASSO 用 $L_1$ 惩罚 $\lambda\sum_j|\beta_j|$，可产生精确零系数；岭回归通常只收缩，不能默认完成离散特征选择。

## 5.5 非线性输入与偏差—方差 {#ml-05-05}

把输入变成 $\phi(x)=(1,x,x^2,\ldots,x^m)$，模型 $f(x)=\boldsymbol\beta^T\phi(x)$ 对输入是多项式，对待估参数仍是线性的，所以最小二乘方法仍适用。高阶多项式可能过拟合，也可能在训练区间外剧烈变化；“训练误差更小”不保证外推更可靠。

平方误差下，设真实响应 $Y=f_*(x)+\epsilon$，噪声条件均值为零、条件方差为 $\sigma^2(x)$，训练集 $D$ 随机。对固定输入 $x$，引入平均预测 $\bar f(x)=E_D\hat f_D(x)$，将误差加减 $\bar f$，展开后利用零均值消去交叉项，得到

$$E_{D,Y}[(Y-\hat f_D(x))^2]=[\bar f(x)-f_*(x)]^2+\operatorname{Var}_D[\hat f_D(x)]+\sigma^2(x).$$

三项分别是偏差平方、估计方差和不可约噪声。岭估计通常引入偏差以降低方差，可能降低总体预测误差，但不保证对所有数据都改善。最小二乘的无偏性还要求固定设计或适当条件下 $E[\boldsymbol\epsilon\mid X]=0$，并不是无需前提的普遍结论。
