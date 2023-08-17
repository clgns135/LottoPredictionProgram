import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일로부터 데이터 읽어오기
data = pd.read_csv("Lotto.csv", header=None, skiprows=1, encoding="utf-8")

# 데이터셋의 숫자 데이터를 문자열에서 숫자로 변환
data = data.iloc[:, :6].apply(pd.to_numeric, errors='coerce')

# 1부터 45까지의 숫자의 출현 빈도 집계
count = np.zeros(45)
for i in range(45):
    count[i] = np.sum(data.values == (i+1))

# 출현 빈도를 확률로 변환하여 정규화
probabilities = count / np.sum(count)

# 높은 순서부터 낮은 순서로 정렬
sorted_indices = np.argsort(probabilities)[::-1]  # 내림차순으로 정렬된 인덱스

# 숫자와 빈도를 포함한 데이터프레임 생성
df = pd.DataFrame({"Number": sorted_indices + 1, "Frequency": probabilities[sorted_indices]})

print("높은 순서부터 낮은 순서:")
for index in sorted_indices:
    print(index + 1)

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.bar(df["Number"], df["Frequency"])
plt.xlabel("Number")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of Numbers")
plt.xticks(np.arange(1, 46, 5))
plt.show()
