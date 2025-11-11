import numpy as np
from feature_extraction import prepare_all_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# creates the list of feature vectors X and the corresponding label list y
X, y = prepare_all_data()

# split the data into training (70%), development (15%) and test set (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=8)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, train_size=0.5, shuffle=True, random_state=8)



print(np.array(y_train).shape)