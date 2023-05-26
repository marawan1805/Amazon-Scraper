import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# Applying EDA to make dataset good for training 
# then training using Naive Bayes classifier 
# and applying cross validation


df = pd.read_csv('titanic.csv')
print('\n')
print(df.tail(10))
print('\n')
for col in df.columns:
    print(col)
print('\n')
print(df.info())
print('\n')
print(df.describe())
print('\n')
plt.show()

#checking how many people survived
sns.countplot(df['Survived'], data=df)
plt.show()

#checking how many people survived by gender
sns.factorplot(x='Survived', col='Sex', kind='count', data=df)
plt.show()
#we can see that more females survived than males

#checking survival rate for each ticket class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.show()
# we can see that richer people travelling in a higher class (Pclass=1)
# survived more than people travelling in a lower class

#chechking the number of companions that came with the passengers
sns.countplot(x='SibSp', data=df)
plt.show()
#most passengers came alone

#df.fillna(df.mean())
#filling null values
print(df.isnull().sum())
print('\n')

#we can drop the columns that won't help in training
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(df.head())
print('\n')

#filling 'age' with mean age
df['Age'].fillna(29, inplace=True)

#filling 'embarked' with mode (bec data is string)
sns.countplot(df['Embarked'], hue=df['Survived'], data=df)
plt.show()
#we can fill with 'S'
df['Embarked'].fillna('S', inplace=True)
print(df.head(10))
print('\n')
print(df.isnull().sum())
print('\n')

#training the data with Naive Bayes Classifier
#first we encode non numerical values
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])
print(df.head())
print('\n')

#feature we're interested in
X = df.drop(['Survived'], axis=1)
y = df['Survived']

#we split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(X_train.dtypes)
print('\n')
print(X_test.isnull().sum())
print('\n')

#scaling according to IQR
cols = X_train.columns
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train.copy(), columns=[cols])
X_test = pd.DataFrame(X_test.copy(), columns=[cols])
print(X_train.head())
print('\n')

#fitting the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

#evaluation metrics
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
y_pred_train = gnb.predict(X_train)
print('\n')
print('Training set score: {:.4f}'.format(gnb.score(X_train, y_train)))
print('\n')
print('Test set score: {:.4f}'.format(gnb.score(X_test, y_test)))
print('\n')

#checking the distribution in test set
print(y_test.value_counts())
print('\n')

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
print('\n')

#visualizing confusion matrix
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

#classification accuracy
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
print('\n')

#classification error
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
print('\n')

#recall
recall = TP / float(TP + FN)
print('Recall : {0:0.4f}'.format(recall))
print('\n')

#applying 10 fold cross validation
scores = cross_val_score(gnb, X_train, y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
print('\n')
print('Average cross-validation score: {:.4f}'.format(scores.mean()))
print('\n')
