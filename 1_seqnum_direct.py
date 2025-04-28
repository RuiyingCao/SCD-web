import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Counter

def filter_and_sort_numbers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    import re
    numbers = [int(num) for num in re.findall(r'\b\d+\b', content)]
    sorted_numbers = sorted(numbers)

    return sorted_numbers

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请提供输入文件路径")
        sys.exit(1)

    file_path = sys.argv[1]
    result = filter_and_sort_numbers(file_path)

    frequency = Counter(result)
    numbers = list(frequency.keys())
    counts = list(frequency.values())

    plt.figure(figsize=(8, 8))
    plt.bar(numbers, counts, color='#263c84', edgecolor='#263c84', linewidth=0.2)
    plt.xlabel('Copy numbers', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.ylim(0, max(counts) * 1.1)


    # 设置坐标轴刻度的字号
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig('uploads/图1.png')
