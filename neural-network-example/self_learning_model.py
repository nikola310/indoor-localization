import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multioutput import MultiOutputRegressor

class SelfLearningModel():

    def __init__(self, model, iter_num, threshold):
        self.__model = model
        self.__iter_num = iter_num
        self.__threshold = threshold

    def set_labeled_dataset(self, labeled_dataset):
        self.__labeled_dataset = labeled_dataset

    def set_unlabeled_dataset(self, unlabeled_dataset):
        self.__unlabeled_dataset = unlabeled_dataset

    def set_features(self, features):
        self.__features = features

    def set_target_variables(self, target_variables):
        self.__target_variables = target_variables

    def train(self):

        self.__model.fit(self.__labeled_dataset[self.__features], self.__labeled_dataset[self.__target_variables])

        predicted_y = self.predict(self.__labeled_dataset[self.__features])
        predicted_y_prob = self.__model.predict_proba(self.__labeled_dataset[self.__features])

        if not getattr(self.__model, 'predict_proba', None):
            self.__plattlr = LogisticRegression()
            preds = self.__model.predict(self.__labeled_dataset[self.__features])
            self.__plattlr.fit(preds.reshape(-1, 1), self.__labeled_dataset[self.__target_variables])

        old_unlabeled_dataset = []
        new_labeled_dataset = self.__labeled_dataset[self.__features]
        old_target_values = self.__labeled_dataset[self.__target_variables]
        it = 0
        while it < self.__iter_num and new_labeled_dataset.shape[0] <= old_target_values.shape[0]:
            # print('Iteration ' + str(it))
            old_unlabeled_dataset = np.copy(predicted_y)

            idx2 = np.where((predicted_y_prob[:, 0] > self.__threshold) | (predicted_y_prob[:, 1] > self.__threshold))[0]
            #numpy.where((unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold))[0]
            new_labeled_dataset = np.vstack((self.__labeled_dataset[self.__features], self.__unlabeled_dataset.iloc[idx2][self.__features]))
            new_target_values = np.hstack((self.__labeled_dataset[self.__target_variables], old_unlabeled_dataset[idx2]))

            self.__model.fit(new_labeled_dataset, new_target_values)
            predicted_y = self.predict(self.__unlabeled_dataset[self.__features])
            predicted_y_prob = self.predict_proba(self.__unlabeled_dataset[self.__features])
            old_target_values = predicted_y
            it += 1


    def predict(self, x):

        return self.__model.predict(x)

    def predict_proba(self, x):

        if getattr(self.__model, 'predict_proba', None):
            return self.__model.predict_proba(x)
        else:
            preds = self.__model.predict(x)
            return self.__plattlr.predict_proba(preds.reshape(-1, 1))

    def get_f1_score(self, x, y):
        return f1_score(y_pred=self.predict(x), y_true=y)*100

    def get_score(self, x, y):
        return accuracy_score(y, self.predict(x))*100