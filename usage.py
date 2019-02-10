import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from klasyfikacja import Klasyfikacja


def main():
    cancer = datasets.load_breast_cancer()
    train_data, test_data, train_target, test_target = train_test_split(
        cancer["data"], cancer["target"], test_size=0.1
    )

    klas = Klasyfikacja(train_data, train_target)
    predicted = klas.predict(test_data)
    acc = klas.score(predicted, test_target, accuracy_score)
    print("Dokładność modelu:", acc)
    acc = klas.score(predicted, test_target, confusion_matrix)
    print("Macierz konfuzji:\n", acc)


if __name__ == "__main__":
    main()
