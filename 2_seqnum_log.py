import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 适用于无界面的服务器环境
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize as NormalizeColor
from matplotlib.cm import ScalarMappable

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取命令行参数，获取文件路径
if len(sys.argv) < 2:
    print("请提供输入文件路径")
    sys.exit(1)

file_name = sys.argv[1]

# 读取文件中的数字
data = []
with open(file_name, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            number = int(line.strip())
            if number > 0:  # 确保数据为正数
                data.append(number)
        except ValueError:
            print(f"无法将行转换为整数: {line.strip()}")

# 检查数据是否为空
if not data:
    print("文件中没有有效的整数数据。")
    sys.exit(1)

# 排序并进行对数变换
data.sort()
log_data = np.log(data)

# 计算对数变换后的直方图数据
counts, bin_edges = np.histogram(log_data, bins=100, density=True)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# 创建自定义颜色映射
colors = [(218 / 255, 227 / 255, 242 / 255), (38 / 255, 60 / 255, 132 / 255)]
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
norm_color = NormalizeColor(vmin=min(counts), vmax=max(counts))
sm = ScalarMappable(cmap=cmap, norm=norm_color)
sm.set_array([])

# 设置图形大小
plt.figure(figsize=(8, 8))

# 绘制直方图
plt.bar(bin_centers, counts, width=np.diff(bin_edges), color=cmap(norm_color(counts)), edgecolor='black', align='center', alpha=0.7)

plt.xlabel('Log.copy number', fontsize=20)
plt.ylabel('Density', fontsize=20)

# 设置坐标轴刻度的字号
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

# 保存图像
plt.savefig('uploads/图2.png', format='png')
