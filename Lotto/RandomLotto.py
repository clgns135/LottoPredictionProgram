import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# CSV 파일로부터 데이터 읽어오기
data = pd.read_csv("Lotto.csv", header=None, skiprows=1, encoding="utf-8")

# 데이터셋의 숫자 데이터를 문자열에서 숫자로 변환
data = data.iloc[:, :7].apply(pd.to_numeric, errors='coerce')

# 추가 특성 생성
data["Difference"] = data.iloc[:, :6].diff(axis=1).iloc[:, 1:].abs().sum(axis=1)  # 숫자 간의 차이 특성
data["Sum"] = data.iloc[:, :6].sum(axis=1)  # 합계 특성
data["Mean"] = data.iloc[:, :6].mean(axis=1)  # 평균 특성

# X, y 데이터 추출
X = data.iloc[:, :9]
y = data.iloc[:, 9]

# 데이터 분할: 학습 데이터와 테스트 데이터로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 열 이름을 문자열로 변환
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# 랜덤 포레스트 회귀 모델 학습
model = RandomForestRegressor(n_estimators=1000, max_depth=5, min_samples_split=5, min_samples_leaf=1, random_state=42)
model.fit(X_train, y_train)

# 다음 숫자 예측
next_numbers = model.predict(X_test).astype(int)

# 다음 숫자에 대한 랜덤 가중치 난수 생성
random_weights = []
for number in next_numbers:
    while True:
        # 각 숫자에 대한 가중치 생성 (예시로 1부터 45까지의 범위에서 랜덤값 사용)
        group = random.sample(range(1, 46), 6)
        if not any(set(group).issubset(set(existing_group)) for existing_group in random_weights):
            random_weights.append(group)
            break

# 생성된 랜덤 가중치 난수 출력 (6개씩 끊어서 출력, 마지막 그룹은 6개가 아니면 출력하지 않음)
for i in range(0, len(random_weights)-5, 6):
    group = random_weights[i:i+6]
    sorted_group = [sorted(nums) for nums in group]
    print(f"{i//6+1}번째 추천: {sorted_group}")



"""
# 중복된 값이 있는지 확인
duplicates = []
for i in range(len(random_weights)-5):
    group = random_weights[i:i+6]
    if any(len(set(x)) < 6 for x in group):
        duplicates.append(i//6+1)
print(duplicates)
"""