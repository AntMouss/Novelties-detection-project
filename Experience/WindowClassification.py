from typing import List
import numpy as np
from scipy import stats
from collections import deque


class WindowClassifierModel:

    def __init__(self, train_similarity_score_samples: List[List], nb_historic: int = 100, classification_centiles: List = None):
        """
        we assume that the similarity_samples have normal distribution

        @param classification_centiles:
        @param nb_historic: number of sample that we want to keep in the historic of the model
        @param train_similarity_score_samples: similarity from the calculator similarity module in percent.
        """
        if classification_centiles is None:
            classification_centiles = np.array([0.02, 0.1, 0.3, 0.6, 0.9, 0.98])
        self.nb_historic = nb_historic
        self.samples_historic = train_similarity_score_samples[:self.nb_historic]
        self.samples_historic = deque(self.samples_historic , maxlen=self.nb_historic)
        self.mean = np.mean(self.samples_historic)
        self.std = np.std(self.samples_historic)
        self.centiles_values = []
        self.groups = [i for i in range(len(classification_centiles) + 1)]
        self.classification_centiles = classification_centiles
        for centile in self.classification_centiles:
            value = stats.norm.ppf(centile, loc=self.mean, scale=self.std)
            self.centiles_values.append(value)


    def __len__(self):

        return len(self.groups)

    def update(self, similarity_score_sample):

        self.samples_historic.pop()
        self.samples_historic.appendleft(similarity_score_sample)
        self.mean = np.mean(self.samples_historic)
        self.std = np.std(self.samples_historic)

    def predict(self, similarity_score_sample):

        for group , centile_value in zip(self.groups , self.centiles_values):
            if similarity_score_sample <= centile_value:
                return group
        return np.nan












