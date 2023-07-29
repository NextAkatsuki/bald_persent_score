from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from bald_csv_edit import dataset
from sklearn.linear_model import LinearRegression

options = dataset[['age', 'gender', 'is_married', 'is_hereditary', 'weight', 'height', 'is_smoker', 'stress']]
bald_prob = dataset['bald_prob']

scaler = StandardScaler()
options_scaled = scaler.fit_transform(options)

model = LinearRegression()

# cross_val_score를 사용하여 교차 검증 수행 (기본적으로 5-fold 교차 검증을 수행)
cv_scores = cross_val_score(model, options_scaled, bald_prob, cv=5)

# 각 폴드의 정확도 출력
print("Cross-validation scores:", cv_scores)

# 평균 정확도 출력
print("Mean Accuracy:", cv_scores.mean())

# 정확도의 표준 편차 출력
print("Accuracy Standard Deviation:", cv_scores.std())
  
