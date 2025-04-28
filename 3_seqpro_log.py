import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 适用于无界面的服务器环境
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
import re
from matplotlib import colors


def filter_and_sort_numbers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    numbers = [int(num) for num in re.findall(r'\b\d+\b', content)]
    return sorted(numbers)


# 读取命令行参数，获取文件路径
if len(sys.argv) < 2:
    print("请提供输入文件路径")
    sys.exit(1)

file_path = sys.argv[1]
PCR10_result = filter_and_sort_numbers(file_path)

# 计算归一化比例
sum_10 = sum(PCR10_result)
norm_10 = [i / sum_10 for i in PCR10_result if i != 0]

# 对数据进行对数变换
log_data1 = np.log(norm_10)

# 创建图形
fig, ax = plt.subplots(figsize=(8, 8))

# 计算直方图
counts, bin_edges = np.histogram(log_data1, bins=50, density=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# 渐变颜色映射
cmap1 = colors.LinearSegmentedColormap.from_list("cmap1", [(218 / 255, 227 / 255, 242 / 255), (38 / 255, 60 / 255, 132 / 255)])
vcenter = np.median(counts)
norm_centered = colors.CenteredNorm(vcenter=vcenter)

# 绘制直方图
bar_container = ax.bar(bin_centers, counts, width=np.diff(bin_edges) * 0.9, color=cmap1(norm_centered(counts)), edgecolor='black', alpha=1.0)

# 计算均值和标准差
mean_log_data = np.mean(log_data1)
std_log_data = np.std(log_data1)
initial_params = [mean_log_data, std_log_data]

# 拟合正态分布
if len(initial_params) == 2 and not any(np.isnan(initial_params)):
    try:
        params, _ = curve_fit(norm.pdf, bin_centers, counts, p0=initial_params, maxfev=10000)
        x_fit = np.linspace(min(log_data1), max(log_data1), 1000)
        y_fit = norm.pdf(x_fit, loc=params[0], scale=params[1])
        fit_line, = ax.plot(x_fit, y_fit, 'red', label=f'norm dist.\nμ={params[0]:.2f}, σ={params[1]:.2f}')
    except Exception as e:
        print(f"Curve fitting failed: {e}")

# 添加均值线
mean_line = ax.axvline(mean_log_data, color='red', linestyle='--', linewidth=2, label='Mean')

# 添加图例
handles = [fit_line, mean_line] if 'fit_line' in locals() else [mean_line]
labels = [h.get_label() for h in handles]
ax.legend(handles=handles, labels=labels, fontsize=14, loc='upper right')

# 设置轴标签
ax.set_xlabel('Log. Proportion of different strands', fontsize=16)
ax.set_ylabel('Density', fontsize=16)

plt.tight_layout()

# 设置坐标轴刻度的字号
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 保存图像
plt.savefig('uploads/图3.png', format='png')
