from typing import List
import numpy as np
from scipy import stats
from collections import deque
import pickle


class WindowClassifierModel:

    inferior_border = 0
    superior_border = 1

    def __init__(self, train_similarity_scores: np.ndarray, nb_historic: int = 100, classification_centiles: List = None):
        """
        we assume that the similarity_samples have normal distribution

        @param classification_centiles:
        @param nb_historic: number of sample that we want to keep in the historic of the model
        @param train_similarity_scores: similarity from the calculator similarity module in percent.
        """
        if classification_centiles is None:
            classification_centiles = np.array([2, 10, 30, 60, 90, 98])
        else:
            classification_centiles = np.array(classification_centiles)
        self.nb_historic = nb_historic
        self.scores_historic = train_similarity_scores[:self.nb_historic]
        self.scores_historic = deque(self.scores_historic, maxlen=self.nb_historic)
        self.mean = np.mean(self.scores_historic)
        self.std = np.std(self.scores_historic)
        self.centiles_values = []
        self.groups = [i + 1 for i in range(len(classification_centiles) + 1)]
        self.classification_centiles = classification_centiles
        for centile in self.classification_centiles / 100:
            value = stats.norm.ppf(centile, loc=self.mean, scale=self.std)
            self.centiles_values.append(value)


    def __len__(self):

        return len(self.groups)

    def update(self, similarity_score):

        if self.inferior_border <= similarity_score <= self.superior_border:
            self.scores_historic.pop()
            self.scores_historic.appendleft(similarity_score)
            self.mean = np.mean(self.scores_historic)
            self.std = np.std(self.scores_historic)
            self.centiles_values = []
            for centile in self.classification_centiles:
                value = stats.norm.ppf(centile, loc=self.mean, scale=self.std)
                self.centiles_values.append(value)


    def predict(self, similarity_score):

        if self.inferior_border <= similarity_score <= self.superior_border:
            for group , centile_value in zip(self.groups , self.centiles_values):
                if similarity_score <= centile_value:
                    return group
            return self.groups[-1]
        else:
            return np.nan

    def print(self , similarity_score):
            predict_group = self.predict(similarity_score)
            if predict_group == 1:
                sup_centile = self.classification_centiles[predict_group-1]
                inf_centile = 0
            elif predict_group == len(self):
                sup_centile = 1
                inf_centile = self.classification_centiles[predict_group-1]
            else:
                sup_centile = self.classification_centiles[predict_group-1]
                inf_centile = self.classification_centiles[predict_group-2]
            print(f"value between centile {inf_centile} and centile {sup_centile}")






    def save(self , path):
        with open(path , "wb") as f:
            f.write(pickle.dumps(self))

    @staticmethod
    def load(path):
        with open(path , "rb") as f:
            classifier = pickle.load(f)
        return classifier




# if __name__ == '__main__':
#
#     path = "/home/mouss/PycharmProjects/novelties-detection-git/model/model.pck"
#     fake_samples = np.random.normal(0.32 , 0.1 , 110)
#     model = WindowClassifierModel(fake_samples)
#     model.save(path)
#     del model
#     model = WindowClassifierModel.load(path)