import math
import sys
from scipy.stats import norm
import numpy as np

# 读取用户上传的文件路径
file_path = sys.argv[1]
R = float(sys.argv[2])
c = float(sys.argv[3])
a = float(sys.argv[4])

# 计算文件行数
with open(file_path, 'r') as file:
    line_count = sum(1 for _ in file)

def calculate_proportions_and_mle(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    total = sum(int(line.strip()) for line in lines)
    sequence_counts = [int(line.strip()) for line in lines]

    # 计算占比
    proportions = [count / total if total != 0 else 0 for count in sequence_counts]

    global mle_mean, mle_variance

    valid_proportions = [p for p in proportions if p > 0]

    if valid_proportions:
        n = len(valid_proportions)
        mle_mean = sum(math.log(p) for p in valid_proportions) / n
        mle_variance = sum((math.log(p) - mle_mean) ** 2 for p in valid_proportions) / n
    else:
        mle_mean = 0
        mle_variance = 0

calculate_proportions_and_mle(file_path)

def f(line_count):
    return math.log(math.log(math.log(line_count)))

z=0.5
xi = norm.ppf(1.0 - 1/R)
scale = np.exp(-mle_mean - mle_variance * xi)
K_L = max(a - z * np.sqrt(a), 0.0) * scale
K_C = a * scale
K_U = (a + z * np.sqrt(a)) * scale

print(f"{round(K_L / line_count, 1)}")
print(f"{round(K_C / line_count, 1)}")
print(f"{round(K_U / line_count, 1)}")
