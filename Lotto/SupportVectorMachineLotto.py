import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

# CSV 파일로부터 데이터 읽어오기
data = pd.read_csv("Lotto.csv", header=None, skiprows=1, encoding="utf-8")

# 데이터셋의 숫자 데이터를 문자열에서 숫자로 변환
data = data.iloc[:, :7].apply(pd.to_numeric, errors='coerce')

# X, y 데이터 추출
X = data.iloc[:, :6]
y = data.iloc[:, 6]

# 데이터 분할: 학습 데이터와 테스트 데이터로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 서포트 벡터 머신 회귀 모델 학습
model = SVR(kernel='rbf', gamma='scale')  # kernel 및 gamma 등의 매개변수 조정
model.fit(X_train, y_train)

# 예측
predicted_numbers = model.predict(X_test)

# 다음 숫자 예측
next_numbers = model.predict(X_test[:5])  # 첫 5개 테스트 데이터를 사용하여 다음 숫자 예측
next_numbers = next_numbers.astype(int)  # 다음 숫자를 정수형으로 변환

# 학습 데이터 정확도 계산
train_accuracy = model.score(X_train, y_train) * 100

# 테스트 데이터 정확도 계산
test_accuracy = model.score(X_test, y_test) * 100

print("다음 숫자 예측:")
for i in range(5):
    print(f"테스트 데이터 {i+1}:")
    print(f"  숫자 6개: {X_test.iloc[i].values}")
    print(f"  보너스 숫자: {next_numbers[i]}")

print(f"\n학습 데이터 정확도: {train_accuracy:.2f}%") # 1.52%
print(f"테스트 데이터 정확도: {test_accuracy:.2f}%") # 1.20%


