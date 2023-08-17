import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV

# CSV 파일로부터 데이터 읽어오기
data = pd.read_csv("Lotto.csv", header=None, skiprows=1, encoding="utf-8")

# 데이터셋의 숫자 데이터를 문자열에서 숫자로 변환
data = data.iloc[:, :7].apply(pd.to_numeric, errors='coerce')

# 추가 특성 생성
data["Difference"] = data.iloc[:, :6].diff(axis=1).iloc[:, 1:].abs().sum(axis=1)  # 숫자 간의 차이 특성
data["Sum"] = data.iloc[:, :6].sum(axis=1)  # 합계 특성
data["Mean"] = data.iloc[:, :6].mean(axis=1)  # 평균 특성
data["Frequency"] = data.iloc[:, :6].apply(lambda row: np.sum(row == row.iloc[5]), axis=1)  # 숫자의 등장 횟수 특성

# X, y 데이터 추출
X = data.iloc[:, :6]
y = data.iloc[:, 6]

# 데이터 분할: 학습 데이터와 테스트 데이터로 나눔
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 열 이름을 문자열로 변환
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# 랜덤 포레스트 모델 정의
model = RandomForestRegressor(random_state=42)

# 탐색할 하이퍼파라미터 범위 정의
param_grid = {
    'n_estimators': [100, 500, 1000],  # 트리 개수
    'max_depth': [5, 10, 15],  # 트리의 최대 깊이
    'min_samples_split': [1, 5, 10],  # 노드를 분할하기 위한 최소 샘플 개수
    'min_samples_leaf': [1, 2, 4]  # 리프 노드에 필요한 최소 샘플 개수
}

# 그리드 서치 수행
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 조합 출력
print("최적의 하이퍼파라미터 조합:")
print(grid_search.best_params_)

# 최적의 모델로 예측 수행
best_model = grid_search.best_estimator_
predicted_numbers = best_model.predict(X_test)
