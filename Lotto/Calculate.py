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

# 학습 데이터 정확도 계산
train_accuracy = model.score(X_train, y_train) * 100

# 테스트 데이터 정확도 계산
test_accuracy = model.score(X_test, y_test) * 100

# 테스트 데이터에 대한 예측 수행
predicted_numbers = model.predict(X_test)

# MSE 계산
mse = mean_squared_error(y_test, predicted_numbers)

# R-squared 계산
r2 = r2_score(y_test, predicted_numbers)

result = f"학습 데이터 정확도: {train_accuracy:.2f}%\n"
result += f"테스트 데이터 정확도: {test_accuracy:.2f}%\n"
result += f"MSE: {mse}\n"
result += f"R-squared: {r2}\n"

print(result)
