# 连续傅里叶级数分解

傅里叶分解的公式描述了如何将一个周期性函数表示为正弦和余弦函数的线性组合。对于一个周期为  $2\pi$  的函数  $f(x)$ ，其傅里叶级数展开可以表示为：

 $f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos(nx) + b_n \sin(nx)) $

其中 \( $a_0$ \)、\( $a_n $\) 和 \( $b_n $\) 分别是 \( $f(x)$ \) 的傅里叶系数，计算公式如下：

 $a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) dx$ 

 $a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) dx$ 

 $b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) dx $

其中  $n$  是非负整数。

这些系数可以通过积分（在连续情况下）或求和（在离散情况下）来计算，从而得到原始函数的傅里叶系数。通过这些系数，我们可以将原始函数表示为一系列正弦和余弦函数的线性组合，这些函数具有不同的频率和振幅。

```python
import numpy as np

# 假设实际的未知复杂函数为f(x)
def f(x):
    return np.sin(x) + 2 * np.cos(x)

# 生成离散值
x_values = np.linspace(0, 2 * np.pi, 1000)

# 计算函数的傅里叶系数
def fourier_coefficients(f, x_values, n):
    coefficients = []
    for i in range(n + 1):
        a_n = 1 / np.pi * np.trapz(f(x_values) * np.cos(i * x_values), x_values)
        b_n = 1 / np.pi * np.trapz(f(x_values) * np.sin(i * x_values), x_values)
        coefficients.append((a_n, b_n))
    return coefficients

# 计算傅里叶系数
N = 10  # 傅里叶级数的阶数
coefficients = fourier_coefficients(f, x_values, N)

# 打印推理出的原函数
def print_reconstructed_function(coefficients):
    reconstructed_function = "f(x) = "
    for i, (a_n, b_n) in enumerate(coefficients):
        if i == 0:
            reconstructed_function += "{:.2f} + ".format(a_n/2)
        else:
            reconstructed_function += "({:.2f} * cos({:.2f}x) + {:.2f} * sin({:.2f}x)) + ".format(a_n, i, b_n, i)
    reconstructed_function = reconstructed_function[:-3]
    print(reconstructed_function)

print("推理出来的原函数：")
print_reconstructed_function(coefficients)
```

# 离散傅里叶级数分解

离散傅里叶级数展开的公式为：

$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j \frac{2\pi}{N} kn} $

其中：
- $x[n]$ 是离散信号序列，在时域上的值。
- $N$ 是信号的长度（采样点数）。
- $X[k]$  是离散傅里叶变换（DFT）的结果，表示信号在频率 $f_k = \frac{k}{N} f_s$  处的复数振幅， $k = 0, 1, 2, ..., N-1$ 。
- $n$ 是时域上的离散时间索引，范围为  $0 \leq n < N$ 。

离散傅里叶级数展开表示了信号  $x[n]$ 在频域上的频率分量，通过对信号进行傅里叶变换，我们可以得到频域上各个频率的振幅，进而实现信号的频域分析。

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义采样频率和信号长度
fs = 100  # 采样频率
N = 1000  # 信号长度
step = 10 # 傅里叶展开阶数
# 生成采样点
t = np.linspace(0, 2*np.pi, N, endpoint=False)  # 0到2pi之间均匀采样

# 计算原始函数在采样点上的值
f = np.sin(t) + 2 * np.cos(t)

# 计算离散傅里叶变换（DFT）
F = np.fft.fft(f)

# 打印恢复的傅里叶级数
print("恢复的傅里叶级数：")
a0 = np.real(F[0]) / N
print("a0 = {:.2f}".format(a0))
for k in range(1, N // 2):
    ak = 2 * np.real(F[k]) / N
    bk = -2 * np.imag(F[k]) / N
    if k < step:
      print("a{} = {:.2f}, b{} = {:.2f}".format(k, ak, k, bk))

# 恢复原始函数
f_restored = np.zeros(N)
f_restored += a0
for k in range(1, N // 2):
    ak = 2 * np.real(F[k]) / N
    bk = -2 * np.imag(F[k]) / N
    f_restored += ak * np.cos(k * t) + bk * np.sin(k * t)

# 绘制原始函数和恢复函数的图像
plt.figure(figsize=(10, 6))
plt.plot(t, f, label='Original Function', color='blue')
plt.plot(t, f_restored, label='Restored Function', linestyle='--', color='red')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original and Restored Functions')
plt.legend()
plt.grid(True)
plt.show()

```

