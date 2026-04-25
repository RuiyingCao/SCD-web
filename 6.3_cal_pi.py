# 6.3_cal_pi.py
import numpy as np


def read_numbers_from_file(filename):
    """从文件中读取数字，每行一个数字，返回 numpy 数组"""
    return np.loadtxt(filename, dtype=float)


def calculate_pi_vectorized(c, r, t):
    """
    向量化计算所有 p_i
    原来是对每个 i 单独计算一次分母，复杂度 O(n^2)
    现在统一先算 weights 和 denominator，复杂度 O(n)
    """
    if c.shape != r.shape:
        raise ValueError("c 和 r 的长度必须相同")

    weights = c * np.power(1.0 + r, t)
    denominator = np.sum(weights)

    if denominator <= 0:
        raise ValueError("分母小于等于 0，无法计算 p_i")

    pi_values = weights / denominator
    return pi_values


def calculate_mu_and_sigma(mean_pi, var_pi):
    """根据给定公式计算 μ(t) 和 σ(t)"""
    mu_t = np.log(mean_pi) - 0.5 * np.log(1 + var_pi / mean_pi**2)
    sigma_t = np.sqrt(np.log(1 + var_pi / mean_pi**2))
    return mu_t, sigma_t


if __name__ == "__main__":
    # 用户指定 t 值——PCR扩增轮数
    t = 10

    # 从文件中读取 c 和 r 值
    c = read_numbers_from_file("synthesisnum.txt")
    r = read_numbers_from_file("PCR_efficiency.txt")

    # 计算 p_i（O(n)）
    pi_values = calculate_pi_vectorized(c, r, t)

    # 计算均值和方差
    mean_pi = np.mean(pi_values)
    var_pi = np.var(pi_values)

    # 计算 μ(t) 和 σ(t)
    mu_t, sigma_t = calculate_mu_and_sigma(mean_pi, var_pi)

    # 如果你以后想保存 p_i，可取消下面注释
    # np.savetxt("cal_pi.txt", pi_values, fmt="%.12f")

    print(
        f"Under your specified conditions, after {t} cycles of PCR amplification, "
        f"the channel probability distribution follows LN({mu_t}, {sigma_t})."
    )
