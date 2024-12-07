import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score
import pickle

file_path = r'FAST-API\BankNote_Authentication.csv'
df = pd.read_csv(file_path)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
#print(score)

pickle_out = open('C:\Kalyani Pakhale\data-science-Interview\FAST-API\model.pkl', 'wb')
pickle.dump(model,pickle_out)
pickle_out.close()