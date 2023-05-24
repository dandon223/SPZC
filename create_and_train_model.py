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

    df = pd.read_csv("MachineLearningCVE/PortScan.csv").drop_duplicates(keep="first")

    int64_columns = []
    f = []

    for i in df.columns[:-1]:
        if df[i].dtype == "int64":
            int64_columns.append(i)
        else:
            f.append(i)

    df[int64_columns] = df[int64_columns].astype("int32")
    df[f] = df[f].astype("float32")
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    x = df.drop([' Label'], axis=1)
    y = df[' Label']

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=0)

    print("X_train, Y_train", X_train.shape, Y_train.shape)
    rus = RandomUnderSampler(random_state=0)
    X_train, Y_train = rus.fit_resample(X_train, Y_train)

    print("X_train, Y_train", X_train.shape, Y_train.shape)

    bayes = GaussianNB()

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    start = time.time()
    bayes.fit(X_train, Y_train)
    print("Time taken to train model: ", time.time()-start," seconds")

    train_predict_X = bayes.predict(X_train)
    train_scores = cross_val_score(bayes, X_train, Y_train, cv=5)
    train_accuracy = metrics.accuracy_score(Y_train, train_predict_X)
    train_confusion_matrix = metrics.confusion_matrix(Y_train, train_predict_X)
    train_classification = metrics.classification_report(Y_train, train_predict_X)

    print_results(Action.TRAIN, train_scores, train_accuracy, train_confusion_matrix, train_classification)

    test_predict_X = bayes.predict(X_test)
    test_scores = cross_val_score(bayes, X_test, Y_test, cv=7)
    test_accuracy = metrics.accuracy_score(Y_test, test_predict_X)
    test_confusion_matrix = metrics.confusion_matrix(Y_test, test_predict_X)
    test_classification = metrics.classification_report(Y_test, test_predict_X)

    print_results(Action.TEST, test_scores, test_accuracy, test_confusion_matrix, test_classification)

    filename = 'model/bayes_model.sav'
    pickle.dump(bayes, open(filename, 'wb'))
