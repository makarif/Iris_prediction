# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv('irisdata.csv')

#Converting words to integer values
df['class_en']=LabelEncoder().fit_transform(df['class'])

X = np.array(df.drop(['class','class_en'], axis=1))
y= np.array(df['class_en'])

X = StandardScaler().fit_transform(X)


#Fitting model with trainig data

clf=DecisionTreeClassifier()
clf.fit(X, y)

# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5, 9, 1.5, 0.2]]))