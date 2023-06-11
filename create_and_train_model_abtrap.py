import pandas as pd
import numpy as np
import time
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from enum import Enum

from sklearn.tree import DecisionTreeClassifier


class Action(Enum):
    TRAIN = "Train"
    TEST = "Test"


def print_results(action: Action, scores, accuracy, confusion_matrix, classification):
    print()
    print('--------------------------- {} Results --------------------------------\n', action.value)
    print("Cross Validation Mean Score: {} \n", scores.mean())
    print()
    print("Model Accuracy: {} \n", accuracy)
    print()
    print("Confusion matrix: {} \n", confusion_matrix)
    print()
    print("Classification report: {} \n", classification)
    print()


if __name__ == "__main__":

    df = pd.read_csv("MachineLearningCVE/ABTRAP_Dataset.csv").drop_duplicates(keep="first")

    int64_columns = []
    float64_columns = []

    for i in df.columns[:-1]:
        if df[i].dtype == "int64":
            int64_columns.append(i)
        else:
            float64_columns.append(i)

    df[int64_columns] = df[int64_columns].astype("int32")
    df[float64_columns] = df[float64_columns].astype("float32")

    x = df.drop(['label'], axis=1)
    y = df['label']

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=120)

    print("X_train, Y_train", X_train.shape, Y_train.shape)
    rus = RandomUnderSampler(random_state=1)
    X_train, Y_train = rus.fit_resample(X_train, Y_train)

    print("X_train, Y_train", X_train.shape, Y_train.shape)

    bayes = GaussianNB()

    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train),
                           columns=imputer.get_feature_names_out())
    X_test = pd.DataFrame(imputer.fit_transform(X_test),
                          columns=imputer.get_feature_names_out())
    clf = DecisionTreeClassifier()
    start = time.time()

    clf.fit(X_train, Y_train)
    print("Time taken to train model: ", time.time()-start, " seconds")

    train_predict_X = clf.predict(X_train)
    train_scores = cross_val_score(clf, X_train, Y_train, cv=5)
    train_accuracy = metrics.accuracy_score(Y_train, train_predict_X)
    train_confusion_matrix = metrics.confusion_matrix(Y_train, train_predict_X)
    train_classification = metrics.classification_report(Y_train, train_predict_X)

    print_results(Action.TRAIN, train_scores, train_accuracy, train_confusion_matrix, train_classification)

    test_predict_X = clf.predict(X_test)
    test_scores = cross_val_score(clf, X_test, Y_test, cv=7)
    test_accuracy = metrics.accuracy_score(Y_test, test_predict_X)
    test_confusion_matrix = metrics.confusion_matrix(Y_test, test_predict_X)
    test_classification = metrics.classification_report(Y_test, test_predict_X)

    print_results(Action.TEST, test_scores, test_accuracy, test_confusion_matrix, test_classification)

    filename = 'model/dt_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
