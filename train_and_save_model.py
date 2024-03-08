# train_and_save_model.py

import numpy as np
from sklearn import datasets, svm
import joblib

def train_and_save_model():
    # Load the iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, :3]
    y = iris.target

    # Create an SVM classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    # Save the trained model
    joblib.dump(clf, 'iris_svm_model.pkl')

if __name__ == "__main__":
    train_and_save_model()
