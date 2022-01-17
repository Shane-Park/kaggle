# 1. 데이터 불러오기

import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# input
train.head()

# output
# PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
# 0            1         0       3  ...   7.2500   NaN         S
# 1            2         1       1  ...  71.2833   C85         C
# 2            3         1       3  ...   7.9250   NaN         S
# 3            4         1       1  ...  53.1000  C123         S
# 4            5         0       3  ...   8.0500   NaN         S

# 2. 데이터 분석

print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
print('----------[train infomation]----------')
print(train.info())
print('----------[test infomation]----------')
print(test.info())

# 3. 그래프로 확인
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()  # setting seaborn default for plots


def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()

    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survivied', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')
    plt.show()


# 각각의 그래프 분석
# pie_chart('Sex')
# pie_chart('Pclass')
# pie_chart('Embarked')

# Bar chart로 시각화
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))


# bar_chart("SibSp")
# bar_chart("Parch")

# 3. 데이터 전 처리 및 특성 추출
train_and_test = [train, test]

# 이름
for dataset in train_and_test:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

print(train.head(5))

## 각각의 타이틀 성별 통계
pd.crosstab(train['Title'], train['Sex'])

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].replace(
        ['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

# 흔하지 않은 Title은 Other로 대체하고 중복되는 표현 통일
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# 추출한 Title 데이터를 학습하기 알맞게 String Data로 변경
for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str)

# 성별 String Data로 변형
for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)

# Embarked 처리
for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)

# 나이 처리
for dataset in train_and_test:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)

# 생존자 나이 밴드 조회
# print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

# 나이별로 String 처리
for dataset in train_and_test:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].map({0: 'Child', 1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'}).astype(str)

# 누락된 pclass3의 요금 평균값 기입
for dataset in train_and_test:
    dataset['Fare'] = dataset['Fare'].fillna(13.675)  # The only one empty fare data's pclass is 3.
# Fare 에도 Binning 적용
for dataset in train_and_test:
    dataset.loc[dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3
    dataset.loc[dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)

# Sibsp & Parch를 Family 로 만들기
for dataset in train_and_test:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
    dataset['Family'] = dataset['Family'].astype(int)
# 특성 추출 및 나머지 전처리
features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand'], axis=1)

# 각각 확인
# print(train.head())
# print(test.head())

# One-hot-encoding for categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop("PassengerId", axis=1).copy()

# scikit-learn 라이브러리 임포트
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle

# 셔플
train_data, train_label = shuffle(train_data, train_label, random_state=5)


# 모델 학습과 평가에 대한 pipe라인
def train_and_test(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction

# 4. 함수에 다섯가지 모델 넣어주면 학습과 평가 완료

# 1) Logistic Regression
log_pred = train_and_test(LogisticRegression())
# 2) SVM
svm_pred = train_and_test(SVC())
# 3) kNN
knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))
# 4) Random Forest
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
# 5) Navie Bayes
nb_pred = train_and_test(GaussianNB())

# 실행 결과
# Accuracy :  82.72 %
# Accuracy :  83.5 %
# Accuracy :  84.51 %
# Accuracy :  88.55 %
# Accuracy :  79.8 %

# 5. 마지막으로 모델을 채택해서 submission.
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "PassengerId": test["PassengerId"],
    "Survived": nb_pred
})

submission.to_csv('output/nb_pred.csv', index=False)
