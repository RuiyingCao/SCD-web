### 计算测序覆盖深度的界限 ###
import math
import sys
from scipy.stats import norm
import numpy as np

# 读取用户上传的文件路径
file_path = sys.argv[1]
R = float(sys.argv[2])
c = float(sys.argv[3])
a = int(sys.argv[4])

# 计算文件行数
with open(file_path, 'r') as file:
    line_count = sum(1 for _ in file)

# 读取 mu 和 sigma
mu_t = float(sys.argv[5])
sigma_t = float(sys.argv[6])

def f(line_count):
    return math.log(math.log(math.log(line_count)))

z=0.5
xi = norm.ppf(1.0 - 1/R)
scale = np.exp(-mu_t - (sigma_t**2) * xi)
K_L = max(a - z * np.sqrt(a), 0.0) * scale
K_C = a * scale
K_U = (a + z * np.sqrt(a)) * scale

print(f"{round(K_L / line_count, 1)}")
print(f"{round(K_C / line_count, 1)}")
print(f"{round(K_U / line_count, 1)}")
