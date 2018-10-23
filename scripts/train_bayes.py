import pickle
from array import array
from pathlib import Path

from sklearn import cross_validation, metrics, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
import numpy
import config
import matplotlib.pyplot as plt
import numpy as np


config = config.get_default_config()

def exists_model():
    model_path = Path(config.get('MODEL_PATH'))
    return model_path.is_file()

# Bestehendes Modell vom Dateisystem einlesen und auf übergebenen Datensatz anwenden
def apply(data):
    print("Modell wird angewendet")
    # Modell laden
    model = joblib.load(config.get('MODEL_PATH'))

    # Eigentliche Daten einlesen und in Features und ids unterteilen
    features2 = np.array(list(data.values()))
    features2 = safe_ln(features2)
    features2 = preprocessing.normalize(features2)
    ids = list(data.keys())
    #print(features2[0])

    predictions = model.predict(features2)
    #print(predictions)
    result = {}

    # Klassifikationen abspeichern
    for index, pred in enumerate(predictions):
        result[ids[index]] = pred

    return result

# Modell anhand eines bestehenden Datensatzes erstellen und auf das Dateisystem abspeichern
def train():
    print("Modell wird erstellt")
    # Trainingsdatensatz einlesen und in Features und Labels unterteilen
    dataset = numpy.loadtxt(config.get('LABEL_PATH') + config.get('LABELED'), delimiter=",")
    features = dataset[:, 0:6]
    features = safe_ln(features)
    features = preprocessing.normalize(features)
    labels = dataset[:, 6]

    cv = KFold(n_splits=5)

    gnb = GaussianNB()

    acc = []

    for train_i, test_i in cv.split(features):
        #print("TRAIN: ", train_i, " TEST: ", test_i)
        #print(features[train_i])
        features_train = features[train_i]
        features_test = features[test_i]
        # features_train, features_test = features[train_i], features[test_i]
        labels_train, labels_test = labels[train_i], labels[test_i]

        # Model erstellen und anhand dessen Klassifikation auf eigentlichem Datensatz vornehmen
        predictions = gnb.fit(features_train, labels_train).predict(features_test)

        print(metrics.accuracy_score(labels_test, predictions))
        print(metrics.precision_score(labels_test, predictions))
        print(metrics.f1_score(labels_test, predictions))
        # Genauigkeit berechnen
        acc.append(metrics.accuracy_score(labels_test, predictions))

        # ROC berechnen
        #        print("AUC: ", metrics.roc_auc_score(labels_test, predictions))

        print(metrics.confusion_matrix(labels_test, predictions))

    print("Std: ", np.std(acc))
    print("Accuracy total: ", np.mean(acc))
    print("Dumb model: ", max(features_test.mean(), 1 - features_test.mean()))

    model = gnb.fit(features, labels)

    # Modell abspeichern
    filename = config.get('MODEL_PATH')
    joblib.dump(model, filename)

    return model


def safe_ln(x, minval=0.0000000001):
    return np.log10(x.clip(min=minval))

def main():
    if exists_model():
        apply()
    else:
        train()


if __name__ == '__main__':
    main()
