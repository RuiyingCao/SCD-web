# 6.3_cal_pi.py
import numpy as np

def read_numbers_from_file(filename):
    """从文件中读取数字，每行一个数字"""
    with open(filename, 'r') as file:
        numbers = [float(line.strip()) for line in file.readlines()]
    return numbers

def calculate_pi(c, r, t, i):
    """根据给定的公式计算 p_i"""
    n = len(c)
    numerator = c[i] * (1 + r[i]) ** t
    denominator = sum(c[j] * (1 + r[j]) ** t for j in range(n))
    pi = numerator / denominator
    return pi

def calculate_mu_and_sigma(mean_pi, var_pi):
    """根据给定的公式计算 μ(t) 和 σ(t)"""
    mu_t = np.log(mean_pi) - 0.5 * np.log(1 + var_pi / mean_pi**2)
    sigma_t = np.sqrt(np.log(1 + var_pi / mean_pi**2))
    return mu_t, sigma_t

# 用户指定 t 值——PCR扩增轮数
t = 10

# 从文件中读取 c 和 r 值
c = read_numbers_from_file("synthesisnum.txt")
r = read_numbers_from_file("PCR_efficiency.txt")

# 检查 c 和 r 的长度是否一致
if len(c) != len(r):
    raise ValueError("c 和 r 的长度必须相同")

# 计算 p_i
pi_values = [calculate_pi(c, r, t, i) for i in range(len(c))]
# 计算均值和方差
mean_pi = np.mean(pi_values)
var_pi = np.var(pi_values)

# 计算 μ(t) 和 σ(t)
mu_t, sigma_t = calculate_mu_and_sigma(mean_pi, var_pi)

'''# 将结果保存到文件
with open("cal_pi.txt", "w") as file:
    for pi in pi_values:
        file.write(f"{pi}\n")'''

print(f"Under your specified conditions, after {t} cycles of PCR amplification, the channel probability distribution follows LN({mu_t}, {sigma_t}).")



