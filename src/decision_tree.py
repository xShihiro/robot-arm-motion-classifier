import numpy as np
from feature_extraction import prepare_all_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# creates the list of feature vectors X and the corresponding label list y
X, y = prepare_all_data()

# split the data into training (70%), development (15%) and test set (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=8)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, train_size=0.5, shuffle=True, random_state=8)

# initialize the decision tree classifier
dtc = DecisionTreeClassifier()

# train the classifier with our training data
dtc.fit(X_train, y_train)

# prediction on the development set
y_dev_pred = dtc.predict(X_dev)

# calculate the accuracy of our dev set prediction
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print(f"\nAccuracy on development set: {dev_accuracy:.2f}")

# detailed classification report
print("\nDetailed classification report:")
print(classification_report(y_dev, y_dev_pred))

# confusion matrix
print("\nConfusion matrix:")
print(confusion_matrix(y_dev, y_dev_pred))