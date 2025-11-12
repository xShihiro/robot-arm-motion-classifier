import numpy as np
from feature_extraction import prepare_all_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# creates the list of feature vectors X and the corresponding label list y
X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_all_data(augment=True, n_augmentations=3)

print(len(X_train))

# initialize the decision tree classifier
dtc = DecisionTreeClassifier()

# train the classifier with our training data
dtc.fit(X_train, y_train)

# prediction on the development set
y_dev_pred = dtc.predict(X_dev)
print(y_dev_pred)

# calculate the accuracy of our dev set prediction
dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print(f"\nAccuracy on development set: {dev_accuracy:.2f}")

# detailed classification report
print("\nDetailed classification report:")
print(classification_report(y_dev, y_dev_pred))

# confusion matrix
print("\nConfusion matrix:")
print(confusion_matrix(y_dev, y_dev_pred))