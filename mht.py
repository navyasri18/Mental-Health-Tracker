import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('mht.data')

X = np.array(df.iloc[:, 0:4])
y = np.array(df.iloc[:, 4:])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
classifier.fit(X_train,y_train)

pickle.dump(classifier, open('abcd.pkl', 'wb'))
