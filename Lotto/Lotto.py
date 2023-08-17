import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
# model = RandomForestRegressor(n_estimators=1000, random_state=42) # 85.19%, -7.04%
model = RandomForestRegressor(n_estimators=1000, max_depth=5, min_samples_split=5, min_samples_leaf=1, random_state=42) # 18.51%, -0.91%
model.fit(X_train, y_train)

# 예측
predicted_numbers = model.predict(X_test)

# 다음 숫자 예측
next_numbers = model.predict(X_test)  # 전체 테스트 데이터를 사용하여 다음 숫자 예측
next_numbers = next_numbers.astype(int)  # 다음 숫자를 정수형으로 변환

# 학습 데이터 정확도 계산
train_accuracy = model.score(X_train, y_train) * 100

# 테스트 데이터 정확도 계산
test_accuracy = model.score(X_test, y_test) * 100

print("다음 숫자 예측:")
for i in range(5):
    print(f"추첨 번호 {i+1}:")
    print(f"  숫자 6개: {X_test.iloc[i, :6].values}" + f"  보너스 숫자: {X_test.iloc[i, 6]}")
    print(f"  추가 특성: 차이={X_test.iloc[i, :6].diff().abs().sum()}, 합={X_test.iloc[i, :6].sum()}, 평균={X_test.iloc[i, :6].mean()}")

#print(f"\n학습 데이터 정확도: {train_accuracy:.2f}%")
#print(f"테스트 데이터 정확도: {test_accuracy:.2f}%")

# 테스트 데이터에 대한 예측 수행
predicted_numbers = model.predict(X_test)

# MSE 계산
mse = mean_squared_error(y_test, predicted_numbers)

# R-squared 계산
r2 = r2_score(y_test, predicted_numbers)

#print(f"MSE: {mse}")
#print(f"R-squared: {r2}")

"""
선형 회귀 모델 - 학습 데이터 정확도 : 4.93%, 테스트 데이터 정확도 : -12.80%
k-최근접 이웃 회귀 모델 - 학습 데이터 정확도 : 16.88%, 테스트 데이터 정확도 : -13.44%
그래디언트 부스팅 회귀 모델 - 학습 데이터 정확도 : 81.58%, 테스트 데이터 정확도 : -39.66%
다층 퍼셉트론 회귀 모델 - 학습 데이터 정확도 : 4.93%, 테스트 데이터 정확도 : -12.80%
랜덤 포레스트 회귀 모델 - 학습 데이터 정확도 : 85.19%, 테스트 데이터 정확도 : -7.04%
서포트 벡터 머신 회귀 모델 - 학습 데이터 정확도 : 1.52%, 테스트 데이터 정확도 : 1.20%
"""