# 7_analysis.py
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，适合服务器环境
import matplotlib.pyplot as plt
import logging
import os

# 设置 Matplotlib 的日志级别为 WARNING 或 ERROR
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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

def calculate_proportions_and_mle(file_path):
    pi_values = []
    with open(file_path, 'r') as file:
        for line in file:
            pi_values.append(float(line.strip()))
    m = len(pi_values)
    K = [i * 0.5 * m for i in range(1, 61)]

    Thm1 = [] # 对数正态
    Thm2 = [] # 均匀
    results2 = []

    # 计算期望
    def formula1(K, pi_values):
        result = sum(1 - np.exp(-K * pi) for pi in pi_values)
        return result

    # 计算方差
    def formula2(K, pi_values):
        sum_exp = sum(pi * np.exp(-K * pi) for pi in pi_values)
        sum_exp_squared = (sum_exp) ** 2
        result = sum(np.exp(-K * pi) * (1 - np.exp(-K * pi)) for pi in pi_values) - K * sum_exp_squared
        return result

    # 对每个K值计算两个公式
    for k in K:
        result1 = formula1(k, pi_values) / m
        result2 = formula2(k, pi_values)
        Thm1.append(round(result1, 3)) # 保存期望
        results2.append(result2) # 保存方差

    def compute_mean_and_variance(m, K):
        mean = m - m * np.exp(-K / m)
        variance = m * (1 - np.exp(-K / m)) * np.exp(-K / m) - K * np.exp(-2 * K / m)
        return mean, variance

    for KK2 in K:
        mean, variance = compute_mean_and_variance(m, KK2)
        Mean = mean / m
        Thm2.append(Mean)

    return Thm1, Thm2, m

def plot_graphs(Thm1, Thm2, preset_coverage=None, expected_proportion=None, coding_redundancy=None):
    x = np.linspace(0.5, 30, 60)

    # 确保 uploads 文件夹存在
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # 绘制 Fig.4-1
    plt.figure(figsize=(24, 6))
    plt.plot(x, Thm2, label='Uniform dist.', color='#a3c0fb', marker='o', linestyle='--', markersize=4)
    plt.plot(x, Thm1, label='simulation dist.', color='#305299', marker='o', markersize=4)
    plt.xlabel('Coverage depth', fontsize=20)
    plt.ylabel('Decoding proportion of encoded strands', fontsize=20)
    plt.xticks(np.arange(0, 30.5, 0.5), fontsize=16)
    plt.yticks(np.arange(0.25, 1.05, 0.05), fontsize=16)
    plt.xlim(0, 30)
    plt.ylim(0.25, 1.05)
    plt.xticks(rotation=45)

    if preset_coverage is not None:
        y = np.interp(preset_coverage, x, Thm1)
        plt.scatter(preset_coverage, y, color='red', marker='*', s=200, label='your setting point')
    elif expected_proportion is not None:
        idx = next((i for i, val in enumerate(Thm1) if val >= expected_proportion), None)
        if idx is not None:
            x_val = x[idx]
            y_val = Thm1[idx]
            plt.scatter(x_val, y_val, color='red', marker='*', s=200, label='your setting point')

    plt.legend(loc='lower right', fontsize=20)

    plt.subplots_adjust(top=1.1)
    plt.savefig(os.path.join(upload_folder, '图7-1.png'), format='png', bbox_inches='tight')
    plt.close()

    # 绘制 Fig.4-2
    uniform = [1 / x for x in Thm2]
    PCR10 = [1 / x for x in Thm1]

    plt.figure(figsize=(24, 6))
    plt.plot(x, uniform, label='Uniform dist.', color='#a3c0fb', marker='o', linestyle='--', markersize=4)
    plt.plot(x, PCR10, label='simulation dist.', color='#305299', marker='o', markersize=4)
    plt.xlabel('Coverage depth', fontsize=20)
    plt.ylabel('Inverse code rate', fontsize=20)
    plt.xticks(np.arange(0, 30.5, 0.5), fontsize=16)
    plt.yticks([1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=16)
    plt.xlim(0, 30)
    plt.ylim(0.95, 4)
    plt.xticks(rotation=45)

    x_val = None
    if coding_redundancy is not None:
        try:
            # 找到 PCR10 曲线上纵坐标值第一个小于或等于 coding_redundancy 的点
            idx = next((i for i, val in enumerate(PCR10) if val <= coding_redundancy), None)
            if idx is not None:
                x_val = x[idx]
                y_val = PCR10[idx]
                plt.scatter(x_val, y_val, color='red', marker='*', s=200, label='your setting point')
                print(f"Marking point at x_val: {x_val}, y_val: {y_val}")  # 增加调试信息
            else:
                print(f"No point found for coding redundancy: {coding_redundancy}")  # 增加调试信息
        except Exception as e:
            print(f"插值计算失败: {str(e)}", file=sys.stderr)  # 增加调试信息
            raise e

    # 调整图例位置，确保它不会遮挡标记点
    plt.legend(loc='upper right', fontsize=20)

    #plt.tight_layout()
    plt.subplots_adjust(top=1.1)
    plt.savefig(os.path.join(upload_folder, '图7-2.png'), format='png', bbox_inches='tight')
    plt.close()

    return x_val


def analyze_and_plot(synthesisnum_file_path, pcr_efficiency_file_path, preset_coverage=None, expected_proportion=None,
                     coding_redundancy=None, t=10):
    # 从文件中读取 c 和 r 值
    c = read_numbers_from_file(synthesisnum_file_path)
    r = read_numbers_from_file(pcr_efficiency_file_path)

    # 检查 c 和 r 的长度是否一致
    if len(c) != len(r):
        raise ValueError(f"c 和 r 的长度必须相同: len(c) = {len(c)}, len(r) = {len(r)}")  # 调试信息

    # 确保 t 是浮点数
    t = float(t)

    # 计算 p_i
    pi_values = [calculate_pi(c, r, t, i) for i in range(len(c))]  # 使用传递的 t 值
    # 计算均值和方差
    mean_pi = np.mean(pi_values)
    var_pi = np.var(pi_values)

    # 计算 μ(t) 和 σ(t)
    mu_t, sigma_t = calculate_mu_and_sigma(mean_pi, var_pi)

    # 删除旧的 cal_pi.txt 文件
    if os.path.exists("cal_pi.txt"):
        os.remove("cal_pi.txt")

    # 将结果保存到文件
    with open("cal_pi.txt", "w") as file:
        for pi in pi_values:
            file.write(f"{pi}\n")

    # 输出 mu_t 和 sigma_t
    print(f"{mu_t}\n{sigma_t}")

    file_path = 'cal_pi.txt'
    Thm1, Thm2, m = calculate_proportions_and_mle(file_path)
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
    parser.add_argument("--t", type=float, default=10, help="Number of PCR amplification rounds.")  # 添加 t 参数

    args = parser.parse_args()

    try:
        Thm1, Thm2, mu_t, sigma_t, x_val = analyze_and_plot(args.synthesisnum_file_path, args.pcr_efficiency_file_path,
                                                            args.preset_coverage, args.expected_proportion,
                                                            args.coding_redundancy, args.t)  # 传递 t 值
        if x_val is not None:
            print(f"x_val: {x_val}")
    except Exception as e:
        print(f"运行 7_analysis.py 失败: {str(e)}", file=sys.stderr)
        sys.exit(1)

