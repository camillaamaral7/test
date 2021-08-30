import pandas as pd
import numpy as np
import sklearn

# --- 1 --- READING THE FILE
df = pd.read_csv("test_data_CANDIDATE.csv", index_col = 0)
#print(df)
#print(df.describe())
# 1.1 Check missing values
#print(df.isnull().sum())

# --- 2 --- FIXING THE DATA
df['sex'] = df['sex'].replace('M', 0)
df['sex'] = df['sex'].replace('F', 1)
df['sex'] = df['sex'].replace('m', 0)
df['sex'] = df['sex'].replace('f', 1)
df=df.replace(np.nan,0)
#print(df)
#print(df.describe())
#print(df.isnull().sum())

# 2.1 - Show all features with quantitative values (not qualitative)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#sns.distplot(df['age'])
#plt.show()
#sns.distplot(df['trestbps'])
#plt.show()
#sns.distplot(df['chol'])
#plt.show()
#sns.distplot(df['thalach'])
#plt.show()
#sns.distplot(df['oldpeak'])
#plt.show()
#sns.distplot(df['trf'])
#plt.show()

# 2.2 - To remove cholesterol error
data_1=df[df['chol']>0]
#sns.distplot(data_1['chol'])
#plt.show()
#print(data_1.describe())
data_cleaned=data_1.reset_index()
#print(data_cleaned)
print(data_cleaned.describe(include='all'))

# --- 3 --- (X = [input], Y = [output])
x = data_cleaned.loc[:200,['age','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','nar','hc','sk','trf']]
y = data_cleaned.loc[:200,['sex']]

x1 = data_cleaned.loc[201:,['age','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','nar','hc','sk','trf']]
y1 = data_cleaned.loc[201:,['sex']] 

# --- 4 --- AI to Sex Predicition
# 4.1 Test train split (recommended for multiple analyzes)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
#print(predictions)

# 4.2 DecisionTreeClassifier (cover both classification and regression)
from sklearn.tree import DecisionTreeClassifier
dtc_clf = DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(x,y)
dtc_prediction = dtc_clf.predict(x1)
#print(dtc_prediction)

# 4.3 Accuracy
#Accuracy Test Train Split
#print(model.score(x_test, y_test))
#Accuracy DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dtc_tree_acc = accuracy_score(dtc_prediction,y1)
#print(dtc_tree_acc)

#--- 5 ---  What is better?
classifiers = ['Test Train Split', 'Decision Tree']
accuracy = np.array([model.score(x_test, y_test),dtc_tree_acc])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc] + ' is the best AI for Sex Prediction')
