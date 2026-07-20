import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sheryanalysis as sh
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
warnings.filterwarnings('ignore')

df = pd.read_csv('./heart.csv')
# print(df.head())
# print(df.shape)
# print(df.duplicated().astype('int')[570])
# print(df['HeartDisease'].value_counts())
# print(df.columns)
df.columns = df.columns.str.lower()
# print(df.head())
# df['heartdisease'].value_counts().plot(kind='bar')
# plt.show()
# print(df.isnull().sum())

# def ploting(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(df[var],kde=True)

# ploting('age',1)
# ploting('restingbp',2)
# ploting('cholesterol',3)
# ploting('maxhr',4)
# plt.show()

ch_mean = df.loc[df['cholesterol'] != 0, 'cholesterol'].mean()
print(ch_mean)
df['cholesterol'] = df['cholesterol'].replace(0,ch_mean)
df['cholesterol'] = df['cholesterol'].round(2)


restingbp = df.loc[df['restingbp'] != 0, 'restingbp'].mean()
# print(restingbp)
df['restingbp'] = df['restingbp'].replace(0,restingbp)
df['restingbp'] = df['restingbp'].round(2)
# print(df.head())

# print(df['restingbp'].value_counts())


# def ploting(var,num):
#     plt.subplot(2,2,num)
#     sns.histplot(df[var],kde=True)

# ploting('age',1)
# ploting('restingbp',2)
# ploting('cholesterol',3)
# ploting('maxhr',4)
# plt.show()


# sh.analyze(df)
# sns.countplot(x = df['sex'])
# plt.show(0)

df_encode = pd.get_dummies(df,drop_first=True)
# print(df_encode.shape)
# print(df_encode.columns)

x = df_encode.drop('heartdisease',axis=1)
y = df_encode['heartdisease']

# print(x.head())
# print(y.head())

x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,train_size=0.20,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

models ={
    "Logistic Regression":LogisticRegression(),
    "KNN":KNeighborsClassifier(),
    "Naive Byeas":GaussianNB(),
    "Decision tree":DecisionTreeClassifier(),
    "SVM":SVC()
}
result = []

for name,model in models.items():
    model.fit(x_train_scaled,y_train)
    y_pridict = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test,y_pridict)
    F1= f1_score(y_test,y_pridict) 
    result.append({
        'Model':name,
        'Accuracy' : round(accuracy,4),
         'F1':round(F1,4)
    })

# print(result)
print(df_encode.columns)


joblib.dump(models['KNN'],'KNN_heart.pkl')
joblib.dump(scaler,'scaler.pkl')
joblib.dump(x.columns.tolist(),'columns.pkl')