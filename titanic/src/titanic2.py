import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# data import
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
gender_submission = pd.read_csv('data/gender_submission.csv')

train.info()

# 성별 numeric 하게 변경
train['Sex_clean'] = train['Sex'].astype('category').cat.codes
test['Sex_clean'] = test['Sex'].astype('category').cat.codes

# Embarked NaN 2개 처리 후 Numeric 데이터로 변경
train['Embarked'].isnull().sum()# 2
test['Embarked'].isnull().sum()# 0
train['Embarked'].value_counts()
# output
# S    644
# C    168
# Q     77

train['Embarked'].fillna('S', inplace=True)
train['Embarked_clean'] = train['Embarked'].astype('category').cat.codes
test['Embarked_clean'] = test['Embarked'].astype('category').cat.codes

# Family 계산 -> SibSp 와 Parch 더해 하나의 Family로 만들기
train['Family'] = 1 + train['SibSp'] + train['Parch']
test['Family'] = 1 + test['SibSp'] + test['Parch']

# Solo -> 혼자인지 여부
train['Solo'] = (train['Family'] == 1)
test['Solo'] = (test['Family'] == 1)

# Fare 5구간으로 Binning
train['FareBin'] = pd.qcut(train['Fare'], 5)
test['FareBin'] = pd.qcut(test['Fare'], 5)

train['FareBin'].value_counts()
# (7.854, 10.5]        184
# (21.679, 39.688]     180
# (-0.001, 7.854]      179
# (39.688, 512.329]    176
# (10.5, 21.679]       172
# Name: FareBin, dtype: int64

# Binding 후에는 Numeric 한 값으로 변경
train['Fare_clean'] = train['FareBin'].astype('category').cat.codes
test['Fare_clean'] = test['FareBin'].astype('category').cat.codes
train['Fare_clean'].value_counts()
# 1    184
# 3    180
# 0    179
# 4    176
# 2    172
# Name: Fare_clean, dtype: int64

# Title 단일화
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

train['Title'].value_counts()
# Mr        517
# Miss      182
# Mrs       125
# Master     40
# Other      23
# Mlle        2
# Ms          1
# Mme         1
# Name: Title, dtype: int64

# Mlle 와 Ms, Mme는 몇개 없으니 단일화
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')

train['Title'].value_counts()
# Mr        517
# Miss      185
# Mrs       126
# Master     40
# Other      23
# Name: Title, dtype: int64

# Test 데이터셋에 대해서도 동일하게 진행
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')

test['Title'].value_counts()
# Mr        240
# Miss       79
# Mrs        72
# Master     21
# Other       6
# Name: Title, dtype: int64

# Title을 Numeric 값으로 변경
train['Title_clean'] = train['Title'].astype('category').cat.codes
test['Title_clean'] = test['Title'].astype('category').cat.codes

# Age 처리 NaN이 많은 컬럼이지만, 중요도가 높기 때문에 pre-processing에 따라 결과가 크게 달라짐.
# Title로 Group화 한 Age의 Median 값으로 채우는 전략.
train['Age'].isnull().sum()
# 177

test['Age'].isnull().sum()
# 86

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

# Age Binning. 5세 단위,  50대는 10세 단위, 60대 이상은 하나의 그룹으로
# Train
train.loc[ train['Age'] <= 10, 'Age_clean'] = 0
train.loc[(train['Age'] > 10) & (train['Age'] <= 16), 'Age_clean'] = 1
train.loc[(train['Age'] > 16) & (train['Age'] <= 20), 'Age_clean'] = 2
train.loc[(train['Age'] > 20) & (train['Age'] <= 26), 'Age_clean'] = 3
train.loc[(train['Age'] > 26) & (train['Age'] <= 30), 'Age_clean'] = 4
train.loc[(train['Age'] > 30) & (train['Age'] <= 36), 'Age_clean'] = 5
train.loc[(train['Age'] > 36) & (train['Age'] <= 40), 'Age_clean'] = 6
train.loc[(train['Age'] > 40) & (train['Age'] <= 46), 'Age_clean'] = 7
train.loc[(train['Age'] > 46) & (train['Age'] <= 50), 'Age_clean'] = 8
train.loc[(train['Age'] > 50) & (train['Age'] <= 60), 'Age_clean'] = 9
train.loc[ train['Age'] > 60, 'Age_clean'] = 10

# Test
test.loc[ test['Age'] <= 10, 'Age_clean'] = 0
test.loc[(test['Age'] > 10) & (test['Age'] <= 16), 'Age_clean'] = 1
test.loc[(test['Age'] > 16) & (test['Age'] <= 20), 'Age_clean'] = 2
test.loc[(test['Age'] > 20) & (test['Age'] <= 26), 'Age_clean'] = 3
test.loc[(test['Age'] > 26) & (test['Age'] <= 30), 'Age_clean'] = 4
test.loc[(test['Age'] > 30) & (test['Age'] <= 36), 'Age_clean'] = 5
test.loc[(test['Age'] > 36) & (test['Age'] <= 40), 'Age_clean'] = 6
test.loc[(test['Age'] > 40) & (test['Age'] <= 46), 'Age_clean'] = 7
test.loc[(test['Age'] > 46) & (test['Age'] <= 50), 'Age_clean'] = 8
test.loc[(test['Age'] > 50) & (test['Age'] <= 60), 'Age_clean'] = 9
test.loc[ test['Age'] > 60, 'Age_clean'] = 10

# Cabin 은 알파뱃만 가져오고, Nan이 많기 때문에 Numeric 값으로 변경 후  pclass로 Group한 median 값 일괄 적용
train['Cabin'].str[:1].value_counts()

# C    59
# B    47
# D    33
# E    32
# A    15
# F    13
# G     4
# T     1
# Name: Cabin, dtype: int64

mapping = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'T': 7
}

train['Cabin_clean'] = train['Cabin'].str[:1]
train['Cabin_clean'] = train['Cabin_clean'].map(mapping)
train['Cabin_clean'] = train.groupby('Pclass')['Cabin_clean'].transform('median')

test['Cabin_clean'] = test['Cabin'].str[:1]
test['Cabin_clean'] = test['Cabin_clean'].map(mapping)
test['Cabin_clean'] = test.groupby('Pclass')['Cabin_clean'].transform('median')

train['Cabin_clean'].value_counts()
# 5.0    491
# 2.0    216
# 4.5    184
# Name: Cabin_clean, dtype: int64

test['Cabin_clean'].value_counts()
# 5.0    311
# 2.0    107
# Name: Cabin_clean, dtype: int64

# Feature 와 Label 정의
feature = [
    'Pclass',
    'SibSp',
    'Parch',
    'Sex_clean',
    'Embarked_clean',
    'Family',
    'Solo',
    'Title_clean',
    'Age_clean',
    'Cabin_clean',
    'Fare_clean',
]

label = [
    'Survived',
]

# HyperParameter

# Cross Validation score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

data = train[feature]
target = train[label]

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=0)
print(cross_val_score(clf, data, target, cv=k_fold, scoring='accuracy', ).mean())
# Accuracy
# 0.8271660424469414

# 예측 결과 파일로 생성
x_train = train[feature]
x_test = test[feature]
y_train = train[label]

clf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=0)
clf.fit(x_train, y_train)
gender_submission['Survived'] = clf.predict(x_test)
gender_submission.to_csv('output/titanic-submission.csv',index=False)