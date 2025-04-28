### 计算测序覆盖深度的界限 ###
import math
import sys

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

upper_bound = math.exp(-mu_t + sigma_t **2 / 2) * (math.log((R - 1) / R)+a*f(line_count)+2*(a+1))
lower_bound = math.exp(-mu_t + sigma_t **2 / 2) * (
    (a - 1) - math.log(2) * math.log((R - 1) / R) + (a - 1) * math.sqrt(
        (-2 * math.log(2) / (a - 1)) * math.log((R - 1) / R)))


# 输出 lower_bound 和 upper_bound
print(f"{round(lower_bound / line_count, 1)}")
print(f"{round(upper_bound / line_count, 1)}")
