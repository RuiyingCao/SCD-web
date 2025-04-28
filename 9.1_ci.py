# 9.1_ci.py
import numpy as np
import sys
import os

# 设置随机种子以获得可重复的结果
np.random.seed(0)


# 打印 sys.argv 以确认传入的参数是否正确
print(f"sys.argv: {sys.argv}")


# 从命令行参数获取 n, x, y
if len(sys.argv) != 4:
    print("Usage: python 9.1_ci.py <n> <x> <y>")
    sys.exit(1)

n = int(sys.argv[1])
x = float(sys.argv[2])
y = float(sys.argv[3])

# 生成n个服从正态分布的数值
normal_numbers = np.random.normal(x, np.sqrt(y), n)

# 将数值四舍五入到最接近的整数
rounded_numbers = np.round(normal_numbers).astype(int)

# 确保 synthesisnum.txt 保存在 uploads 目录下
uploads_folder = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_folder, exist_ok=True)
output_file = os.path.join(uploads_folder, "synthesisnum.txt")

# 将生成的正整数保存到文件中，每个数一行
with open(output_file, "w") as f:
    for num in rounded_numbers:
        f.write(f"{num}\n")

print(f"synthesisnum.txt 保存路径: {output_file}")  # 调试信息
