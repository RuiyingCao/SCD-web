# 7_analysis.py
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')  # 适用于无界面的服务器环境
import matplotlib.pyplot as plt
import logging
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置 Matplotlib 的日志级别为 WARNING 或 ERROR
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def read_numbers_from_file(filename):
    """从文件中读取数字，每行一个数字，返回 numpy 数组"""
    return np.loadtxt(filename, dtype=float)


def calculate_pi_vectorized(c, r, t):
    """
    向量化计算所有 p_i
    原实现对每个 i 都重新计算一次分母，复杂度 O(n^2)
    当前实现先统一计算 weights 和 denominator，复杂度 O(n)
    """
    if c.shape != r.shape:
        raise ValueError(f"c 和 r 的长度必须相同: len(c) = {len(c)}, len(r) = {len(r)}")

    weights = c * np.power(1.0 + r, t)
    denominator = np.sum(weights)

    if denominator <= 0:
        raise ValueError("分母小于等于 0，无法计算 p_i")

    pi_values = weights / denominator
    return pi_values


def calculate_mu_and_sigma(mean_pi, var_pi):
    """根据给定的公式计算 μ(t) 和 σ(t)"""
    mu_t = np.log(mean_pi) - 0.5 * np.log(1 + var_pi / mean_pi**2)
    sigma_t = np.sqrt(np.log(1 + var_pi / mean_pi**2))
    return mu_t, sigma_t


def calculate_proportions_and_mle_from_pi(pi_values):
    """
    直接基于 pi_values 计算后续理论曲线
    不再先写 cal_pi.txt 再读回来
    """
    pi_values = np.asarray(pi_values, dtype=float)
    m = len(pi_values)
    K = np.array([i * 0.5 * m for i in range(1, 61)], dtype=float)

    Thm1 = []  # 非均匀
    Thm2 = []  # 均匀
    results2 = []

    # 计算期望
    def formula1(k, pi_vals):
        return np.sum(1.0 - np.exp(-k * pi_vals))

    # 计算方差
    def formula2(k, pi_vals):
        exp_term = np.exp(-k * pi_vals)
        sum_exp = np.sum(pi_vals * exp_term)
        return np.sum(exp_term * (1.0 - exp_term)) - k * (sum_exp ** 2)

    for k in K:
        result1 = formula1(k, pi_values) / m
        result2 = formula2(k, pi_values)
        Thm1.append(round(float(result1), 3))
        results2.append(float(result2))

    def compute_mean_and_variance_uniform(m_val, k_val):
        mean = m_val - m_val * np.exp(-k_val / m_val)
        variance = m_val * (1 - np.exp(-k_val / m_val)) * np.exp(-k_val / m_val) - k_val * np.exp(-2 * k_val / m_val)
        return mean, variance

    for k_val in K:
        mean, variance = compute_mean_and_variance_uniform(m, k_val)
        Thm2.append(float(mean / m))

    return Thm1, Thm2, m


def plot_graphs(Thm1, Thm2, preset_coverage=None, expected_proportion=None, coding_redundancy=None):
    x = np.linspace(0.5, 30, 60)

    # 确保 uploads 文件夹存在
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # 绘制 Fig.7-1
    plt.figure(figsize=(24, 6))
    plt.plot(x, Thm2, label='均匀分布', color='#a3c0fb', marker='o', linestyle='--', markersize=4)
    plt.plot(x, Thm1, label='模拟分布', color='#305299', marker='o', markersize=4)
    plt.xlabel('测序覆盖深度', fontsize=26)
    plt.ylabel('编码链解码比例', fontsize=26)
    plt.xticks(np.arange(0, 30.5, 0.5), fontsize=20)
    plt.yticks(np.arange(0.25, 1.05, 0.05), fontsize=20)
    plt.xlim(0, 30)
    plt.ylim(0.25, 1.05)
    plt.xticks(rotation=45)

    if preset_coverage is not None:
        y = np.interp(preset_coverage, x, Thm1)
        plt.scatter(preset_coverage, y, color='red', marker='*', s=200, label='用户设定点')
    elif expected_proportion is not None:
        idx = next((i for i, val in enumerate(Thm1) if val >= expected_proportion), None)
        if idx is not None:
            x_val = x[idx]
            y_val = Thm1[idx]
            plt.scatter(x_val, y_val, color='red', marker='*', s=200, label='用户设定点')

    plt.legend(loc='lower right', fontsize=24)
    plt.subplots_adjust(top=1.1)
    plt.savefig(os.path.join(upload_folder, '图7-1.png'), format='png', bbox_inches='tight')
    plt.close()

    # 绘制 Fig.7-2
    uniform = [1 / val for val in Thm2]
    simulated = [1 / val for val in Thm1]

    plt.figure(figsize=(24, 6))
    plt.plot(x, uniform, label='均匀分布', color='#a3c0fb', marker='o', linestyle='--', markersize=4)
    plt.plot(x, simulated, label='模拟分布', color='#305299', marker='o', markersize=4)
    plt.xlabel('测序覆盖深度', fontsize=26)
    plt.ylabel('码率倒数', fontsize=26)
    plt.xticks(np.arange(0, 30.5, 0.5), fontsize=20)
    plt.yticks([1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=20)
    plt.xlim(0, 30)
    plt.ylim(0.95, 4)
    plt.xticks(rotation=45)

    x_val = None
    if coding_redundancy is not None:
        try:
            idx = next((i for i, val in enumerate(simulated) if val <= coding_redundancy), None)
            if idx is not None:
                x_val = x[idx]
                y_val = simulated[idx]
                plt.scatter(x_val, y_val, color='red', marker='*', s=200, label='用户设定点')
                print(f"Marking point at x_val: {x_val}, y_val: {y_val}")
            else:
                print(f"No point found for coding redundancy: {coding_redundancy}")
        except Exception as e:
            print(f"插值计算失败: {str(e)}", file=sys.stderr)
            raise e

    plt.legend(loc='upper right', fontsize=24)
    plt.subplots_adjust(top=1.1)
    plt.savefig(os.path.join(upload_folder, '图7-2.png'), format='png', bbox_inches='tight')
    plt.close()

    return x_val


def analyze_and_plot(synthesisnum_file_path, pcr_efficiency_file_path,
                     preset_coverage=None, expected_proportion=None,
                     coding_redundancy=None, t=10):
    # 从文件中读取 c 和 r 值
    c = read_numbers_from_file(synthesisnum_file_path)
    r = read_numbers_from_file(pcr_efficiency_file_path)

    # 确保 t 是浮点数
    t = float(t)

    # 计算 p_i（O(n)）
    pi_values = calculate_pi_vectorized(c, r, t)

    # 计算均值和方差
    mean_pi = float(np.mean(pi_values))
    var_pi = float(np.var(pi_values))

    # 计算 μ(t) 和 σ(t)
    mu_t, sigma_t = calculate_mu_and_sigma(mean_pi, var_pi)

    # 如果你还想保留 cal_pi.txt 供调试/检查，可以继续保存
    if os.path.exists("cal_pi.txt"):
        os.remove("cal_pi.txt")
    np.savetxt("cal_pi.txt", pi_values, fmt="%.12f")

    # 输出 mu_t 和 sigma_t
    print(f"{mu_t}\n{sigma_t}")

    # 直接用 pi_values 算图，不再二次读文件
    Thm1, Thm2, m = calculate_proportions_and_mle_from_pi(pi_values)
    x_val = plot_graphs(Thm1, Thm2, preset_coverage, expected_proportion, coding_redundancy)

    return Thm1, Thm2, mu_t, sigma_t, x_val


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze sequencing data.")
    parser.add_argument("synthesisnum_file_path", help="Path to the synthesis number file.")
    parser.add_argument("pcr_efficiency_file_path", help="Path to the PCR efficiency file.")
    parser.add_argument("--preset_coverage", type=float, help="Preset coverage depth.")
    parser.add_argument("--expected_proportion", type=float, help="Expected decoding proportion of encoded strands.")
    parser.add_argument("--coding_redundancy", type=float, help="Coding redundancy.")
    parser.add_argument("--t", type=float, default=10, help="Number of PCR amplification rounds.")

    args = parser.parse_args()

    try:
        Thm1, Thm2, mu_t, sigma_t, x_val = analyze_and_plot(
            args.synthesisnum_file_path,
            args.pcr_efficiency_file_path,
            args.preset_coverage,
            args.expected_proportion,
            args.coding_redundancy,
            args.t
        )
        if x_val is not None:
            print(f"x_val: {x_val}")
    except Exception as e:
        print(f"运行 7_analysis.py 失败: {str(e)}", file=sys.stderr)
        sys.exit(1)
