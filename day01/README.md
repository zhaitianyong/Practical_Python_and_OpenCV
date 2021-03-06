使用大津算法来二值化图像吧。

大津算法，也被称作最大类间方差法，是一种可以自动确定二值化中阈值的算法。

从类内方差和类间方差的比值计算得来：

小于阈值$t$的类记作$0$，大于阈值$t$的类记作$1$；
$w_0$和$w_1$是被阈值$t$分开的两个类中的像素数占总像素数的比率（满足$w_0+w_1=1$）；
${S_0}^2$， ${S_1}^2$是这两个类中像素值的方差；
$M_0$，$M_1$是这两个类的像素值的平均值；
即：

类内方差：${S_w}^2=w_0\ {S_0}^2+w_1\ {S_1}^2$
类间方差：${S_b}^2 = w_0 \ (M_0 - M_t)^2 + w_1\ (M_1 - M_t)^2 = w_0\ w_1\ (M_0 - M_1) ^2$
图像所有像素的方差：${S_t}^2 = {S_w}^2 + {S_b}^2 = \text{常数}$
根据以上的式子，我们用以下的式子计算分离度$X$：^1

$$ X = \frac{{S_b}^2}{{S_w}^2} = \frac{{S_b}^2}{{S_t}^2 - {S_b}^2} $$

也就是说： $$ \arg\max\limits_{t}\ X=\arg\max\limits_{t}\ {S_b}^2 $$ 换言之，如果使${S_b}^2={w_0}\ {w_1}\ (M_0 - M_1)^2$最大，就可以得到最好的二值化阈值$t$。