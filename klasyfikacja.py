import sklearn
from sklearn.linear_model import LogisticRegression


class Klasyfikacja(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target

        self.logistic_regression = LogisticRegression()
        self.logistic_regression.fit(self.data, self.target)

    def predict(self, data):
        predicted = self.logistic_regression.predict(data)
        return predicted

    def score(self, predicted, expected, score_fn):
        accuracy = score_fn(expected, predicted)
        return accuracy
