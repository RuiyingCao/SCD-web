import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，适合服务器环境
import matplotlib.pyplot as plt
from sympy import symbols, exp, lambdify

def calculate_proportions_and_mle(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total = 0
    sequence_counts = []
    for line in lines:
        count = int(line.strip())
        sequence_counts.append(count)
        total += count

    proportions = [count / total if total != 0 else 0 for count in sequence_counts]
    valid_proportions = [p for p in proportions if p > 0]

    if valid_proportions:
        n = len(valid_proportions)
        mle_mean = sum(np.log(p) for p in valid_proportions) / n
        mle_variance = sum((np.log(p) - mle_mean) ** 2 for p in valid_proportions) / n
    else:
        mle_mean = 0
        mle_variance = 0

    with open(output_file, 'w', encoding='utf-8') as output:
        for proportion in proportions:
            output.write(f"{proportion:.6f}\n")

    m = len(sequence_counts)
    K = [i * 0.5 * m for i in range(1, 61)]
    Thm1 = []
    Thm2 = []

    for ele in K:
        E_lst = []
        Var_lst = []
        for i in range(1, 11):
            E = 0
            Var1 = 0
            Var2 = 0
            Var3 = 0
            s = np.random.lognormal(mean=mle_mean, sigma=mle_variance, size=m)
            positive_numbers = s[s > 0]
            sampled_data = positive_numbers
            for j in range(m):
                E += 1 - np.exp(-ele * sampled_data[j])
                Var1 += (1 - np.exp(-ele * sampled_data[j])) * (np.exp(-ele * sampled_data[j]))
                Var2 += sampled_data[j] * (np.exp(-ele * sampled_data[j]))
            Var3 = Var1 - ele * ((Var2) ** 2)
            E_lst.append(E)
            Var_lst.append(Var3)
        avg = np.mean(E_lst)
        avg_pro = avg / m
        Thm1.append(round(avg_pro, 3))

    def compute_mean_and_variance(m, K):
        mean = m - m * np.exp(-K / m)
        variance = m * (1 - np.exp(-K / m)) * np.exp(-K / m) - K * np.exp(-2 * K / m)
        return mean, variance

    for KK2 in K:
        mean, variance = compute_mean_and_variance(m, KK2)
        Mean = mean / m
        Thm2.append(Mean)

    return Thm1, Thm2, mle_mean, mle_variance, m

def find_nearest_index(arr, target):
    idx = np.abs(np.array(arr) - target).argmin()
    return idx

def plot_graphs(Thm1, Thm2, mle_mean, mle_variance, m, R, preset_coverage, expected_completeness, mark_option):
    x = np.linspace(0.5, 30, 60)

    # 绘制 Fig.4-1
    plt.figure(figsize=(24, 6))
    plt.plot(x, Thm2, label='Uniform dist.', color='#a3c0fb', marker='o', linestyle='--', markersize=4)
    plt.plot(x, Thm1, label='channel dist. of your sequencing data', color='#305299', marker='o', markersize=4)
    plt.xlabel('Coverage depth', fontsize=20)
    plt.ylabel('Decoding proportion of encoded strands', fontsize=20)
    plt.xticks(np.arange(0, 30.5, 0.5), fontsize=16)
    plt.yticks(np.arange(0.25, 1.05, 0.05), fontsize=16)
    plt.xlim(0, 30)
    plt.ylim(0.25, 1.05)
    plt.xticks(rotation=45)

    coordinate_info_4_1 = ""
    if mark_option == 'preset_coverage':
        # 标注 preset_coverage 对应的点
        idx = find_nearest_index(x, preset_coverage)
        x_preset = x[idx]
        y_preset = Thm1[idx]
        plt.plot(x_preset, y_preset, 'r*', markersize=15, label='your setting point')
        coordinate_info_4_1 = f"Preset sequencing coverage depth: {x_preset:.2f}, expected decoding proportion of encoded strands: {y_preset:.2f}"
    elif mark_option == 'expected_completeness':
        # 标注 expected_completeness 对应的第一个大于该值的点
        idx = np.argmax(np.array(Thm1) > expected_completeness)
        x_expected = x[idx]
        y_expected = Thm1[idx]
        plt.plot(x_expected, y_expected, 'r*', markersize=15, label='your setting point')
        coordinate_info_4_1 = f"Expected sequencing coverage depth: {x_expected:.2f}"

    plt.legend(loc='lower right', fontsize=20)
    plt.subplots_adjust(top=1.1)
    plt.savefig('uploads/图4-1.png', format='png', bbox_inches='tight')
    plt.close()

    # 绘制 Fig.4-2
    uniform = [1 / x for x in Thm2]
    PCR10 = [1 / x for x in Thm1]

    plt.figure(figsize=(12, 8))
    plt.plot(x, uniform, label='Uniform dist.', color='#a3c0fb', marker='o', linestyle='--', markersize=4)
    plt.plot(x, PCR10, label='channel dist. of your sequencing data', color='#305299', marker='o', markersize=4)
    plt.xlabel('Coverage depth', fontsize=20)
    plt.ylabel('Inverse code rate', fontsize=20)
    plt.xticks(np.arange(0, 10.5, 0.5), fontsize=16)
    plt.yticks([1, 1.5, 2, 2.5, 3, 3.5, 4], fontsize=16)
    plt.xlim(0, 10)
    plt.ylim(0.95, 4)

    # 标注 R 对应的点
    if R is not None:
        R_value = R
        idx = find_nearest_index(PCR10, R_value)
        x_R = x[idx]
        plt.plot(x_R, R_value, 'r*', markersize=15, label='your setting point')
        coordinate_info_4_2 = f"Expected Sequencing Coverage Depth: {x_R:.2f}"  # 更新 coordinate_info 为 Fig.4-2 的坐标信息
    else:
        coordinate_info_4_2 = ""

    plt.legend(loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig('uploads/图4-2.png', format='png', bbox_inches='tight')
    plt.close()

    # 绘制 Fig.4-3
    K_sym, p_sym, m_sym = symbols('K p m')
    f_expr = exp(-p_sym * K_sym) - exp(-2 * p_sym * K_sym) - (m_sym / K_sym) * (p_sym * K_sym) ** 2 * exp(-2 * p_sym * K_sym)
    f_func = lambdify((K_sym, p_sym, m_sym), f_expr, 'numpy')

    with open('4_seqnum_pro.txt', 'r') as file:
        content = file.read()
        numbers = [float(num) for num in content.split()]
        total = sum(numbers)
        count = len(numbers)
        average = total / count

    K_values = np.linspace(0.01, 10 * m, 400)  # 限制K_values的范围以避免溢出
    f_values = f_func(K_values, average, m)

    plt.figure(figsize=(12, 8))
    plt.plot(K_values / m, f_values, label='Variance curve', color='#263C84', linestyle='-')
    plt.ylim(bottom=-0.00001)
    plt.xticks(np.arange(0, 11, 1))
    plt.xlabel('Coverage depth', fontsize=20)
    plt.ylabel("Variance per strand (var/n)", fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.tight_layout()

    # 设置坐标轴刻度的字号
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('uploads/图4-3.png', format='png', bbox_inches='tight')
    plt.close()

    # 返回两个坐标信息
    return coordinate_info_4_1, coordinate_info_4_2

def main(file_path, R, preset_coverage, expected_completeness, mark_option):
    output_file = "4_seqnum_pro.txt"
    Thm1, Thm2, mle_mean, mle_variance, line_count = calculate_proportions_and_mle(file_path, output_file)
    coordinate_info_4_1, coordinate_info_4_2 = plot_graphs(Thm1, Thm2, mle_mean, mle_variance, line_count, R, preset_coverage, expected_completeness, mark_option)
    return coordinate_info_4_1, coordinate_info_4_2  # 返回坐标信息

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("请提供输入文件路径、编码冗余度 R、预设覆盖深度 preset_coverage、期望完整性 expected_completeness 和标记选项 mark_option")
        sys.exit(1)

    file_path = sys.argv[1]
    R = float(sys.argv[2])
    preset_coverage = float(sys.argv[3])
    expected_completeness = float(sys.argv[4])
    mark_option = sys.argv[5]
    coordinate_info_4_1, coordinate_info_4_2 = main(file_path, R, preset_coverage, expected_completeness, mark_option)
    print(coordinate_info_4_1)  # 打印 Fig.4-1 的坐标信息
    print(coordinate_info_4_2)  # 打印 Fig.4-2 的坐标信息

