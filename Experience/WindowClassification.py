from typing import List
import numpy as np
from scipy import stats
from collections import deque


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
            classification_centiles = np.array([0.02, 0.1, 0.3, 0.6, 0.9, 0.98])
        self.nb_historic = nb_historic
        self.scores_historic = train_similarity_scores[:self.nb_historic]
        self.scores_historic = deque(self.scores_historic, maxlen=self.nb_historic)
        self.mean = np.mean(self.scores_historic)
        self.std = np.std(self.scores_historic)
        self.centiles_values = []
        self.groups = [i + 1 for i in range(len(classification_centiles) + 1)]
        self.classification_centiles = classification_centiles
        for centile in self.classification_centiles:
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

