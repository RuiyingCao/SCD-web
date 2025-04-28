# 6.1.1_generate_ci.py
import numpy as np
import sys
import os

# 设置随机种子以获得可重复的结果
np.random.seed(0)

# 定义均值和方差
x = float(sys.argv[1])  # 每种序列期望条数
y = float(sys.argv[2])   # 方差

# 生成n个服从正态分布的数值
file_path = sys.argv[3]  # 替换为你的txt文件路径
with open(file_path, 'r', encoding='utf-8') as file:
    line_count = sum(1 for line in file)
n = line_count  # 共多少种序列
print(f"PCR_efficiency.txt 路径: {file_path}, 行数: {n}")  # 调试信息

normal_numbers = np.random.normal(x, np.sqrt(y), n)

# 将数值四舍五入到最接近的整数
rounded_numbers = np.round(normal_numbers).astype(int)

# 确保 synthesisnum.txt 保存在 uploads 目录下
uploads_folder = os.path.dirname(file_path)  # 使用上传文件所在的目录作为 uploads 目录
synthesisnum_file_path = os.path.join(uploads_folder, "synthesisnum.txt")

# 确保 uploads 目录存在
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)

# 将生成的正整数保存到文件中，每个数一行
with open(synthesisnum_file_path, "w") as f:
    for num in rounded_numbers:
        f.write(f"{num}\n")

print(f"synthesisnum.txt 保存路径: {synthesisnum_file_path}")  # 调试信息
