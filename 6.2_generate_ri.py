import numpy as np
import sys
import os

# 设置随机种子以获得可重复的结果
np.random.seed(0)

# 打印 sys.argv 以确认传入的参数是否正确
print(f"sys.argv: {sys.argv}")

# 从命令行参数获取 n, x, y
if len(sys.argv) != 4:
    print("Usage: python 6.2_generate_ri.py <n> <x> <y>")
    sys.exit(1)

n = int(sys.argv[1])
x = float(sys.argv[2])
y = float(sys.argv[3])

# 初始化 positive_numbers 数组
positive_numbers = np.array([])

# 生成 n 个在 [0.85, 1.05] 范围内的正态分布随机数
while len(positive_numbers) < n:
    # 生成一批正态分布随机数
    batch_size = n - len(positive_numbers)
    normal_numbers = np.random.normal(x, np.sqrt(y), batch_size)

    # 裁剪到 [0.85, 1.05] 范围内
    clipped_numbers = np.clip(np.abs(normal_numbers), 0.85, 1.05)

    # 过滤掉不在 [0.85, 1.05] 范围内的数值
    valid_numbers = clipped_numbers[(clipped_numbers >= 0.85) & (clipped_numbers <= 1.05)]

    # 将有效的数值添加到 positive_numbers 中
    positive_numbers = np.concatenate((positive_numbers, valid_numbers))

# 打印生成的随机数的统计信息
print(f"生成的随机数均值: {np.mean(positive_numbers):.4f}")
print(f"生成的随机数标准差: {np.std(positive_numbers):.4f}")

# 确保 PCR_efficiency.txt 保存在 uploads 目录下
uploads_folder = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_folder, exist_ok=True)
output_file = os.path.join(uploads_folder, "PCR_efficiency.txt")

# 将生成的正数保存到文件中，每个数一行
with open(output_file, "w") as f:
    for num in positive_numbers:
        f.write(f"{num:.4f}\n")  # 保留4位小数

print(f"PCR_efficiency.txt 保存路径: {output_file}")  # 调试信息
