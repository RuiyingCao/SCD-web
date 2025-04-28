# 9.2_ci.py
import numpy as np
import sys
import os

def generate_numbers(n, S, output_file):
    # 生成n个服从正态分布的随机数
    numbers = np.random.normal(loc=S / n, scale=S / (n * 3), size=n)

    # 调整随机数使其和为S
    numbers = numbers - np.mean(numbers) + S / n

    # 转换为正整数
    numbers = np.abs(numbers)
    numbers = np.round(numbers).astype(int)

    # 如果调整后总和不等于S，进行微调
    while np.sum(numbers) != S:
        diff = S - np.sum(numbers)
        if diff > 0:
            # 如果总和小于S，随机增加一些数
            for _ in range(diff):
                idx = np.random.randint(0, n)
                numbers[idx] += 1
        elif diff < 0:
            # 如果总和大于S，随机减少一些数
            for _ in range(-diff):
                idx = np.random.randint(0, n)
                if numbers[idx] > 0:  # 确保不减少到负数
                    numbers[idx] -= 1

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as file:
        for num in numbers:
            file.write(f"{num}\n")

# 从命令行参数获取 n, S
if len(sys.argv) != 3:
    print("Usage: python 9.2_ci.py <n> <S>")
    sys.exit(1)

n = int(sys.argv[1])
S = int(sys.argv[2])

# 确保 synthesisnum.txt 保存在 uploads 目录下
uploads_folder = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_folder, exist_ok=True)
output_file = os.path.join(uploads_folder, "synthesisnum.txt")

# 调用函数生成并写入文件
generate_numbers(n, S, output_file)

print(f"synthesisnum.txt 保存路径: {output_file}")  # 调试信息
