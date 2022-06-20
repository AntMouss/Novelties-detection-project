import numpy as np
import copy
from data_utils import ExperiencesResults , Alerte
from scipy.stats import ttest_ind



class Sampler:

    def __init__(self , results : ExperiencesResults ):

        self.results = results.results
        self.info = results.info

    @property
    def samples(self):

        topic_samples = {
            "before": [],
            "after": [],
            "middle": [],
            "in": [],
            "out": [],
            "inside": []
        }
        samples = [copy.deepcopy(topic_samples) for _ in range(self.info["nb_topics"])]
        for result in self.results:
            similarity = result.similarity
            for i , (topic_res_w , topic_res_wout) in enumerate(zip(similarity['with'] , similarity['without'])):
                tmp = np.abs(np.array(topic_res_w) - np.array(topic_res_wout))
                for j , value in enumerate(tmp):
                    key = Sampler.choose_key(j, result.metadata.ranges)
                    samples[i][key].append(value)
        return samples


    @staticmethod
    def choose_key(idx_window , ranges):

        if idx_window < ranges[0][0]:
            return 'before'
        elif idx_window > ranges[-1][1]:
            return 'after'
        else:
            for entry , out in ranges:
                if idx_window == out:
                    return 'out'
                elif idx_window == entry:
                    return 'in'
                elif entry < idx_window < out:
                    return 'inside'
            return 'middle'


class Analyser:

    def __init__(self , samples , risk = 0.05 , trim = 0):
        self.trim = trim
        self.risk = risk
        self.samples = samples
        self.nb_topics = len(samples)
        self.types_window = list(samples[0].keys())


    @property
    def matrix(self):
        matrix = []
        for topic_id in range(self.nb_topics):
            matrix.append(self.topic_pvalue_matrix(topic_id , trim=self.trim))
        return matrix


    def generate_alert(self, topic_id, type_window1, type_window2):
        if type_window1 not in self.types_window and type_window2 not in self.types_window:
            raise Exception(f"this windows type don't exist , here is "
                            f"the differents type:\n'{self.types_window}'")
        idx1 = self.types_window.index(type_window1)
        idx2 = self.types_window.index(type_window2)
        pvalue = self.matrix[topic_id][idx1][idx2]
        if pvalue < self.risk:
            return Alerte(topic_id , risk=self.risk , windows=[type_window1 , type_window2] , pvalue= pvalue)


    def test_hypothesis(self):

        target_window_types = ['in' , 'out']
        other_window_types = ['inside' , 'middle' , 'after' , 'before']
        for topic_id in range (self.nb_topics):
            for target_window in target_window_types:
                for other_window in other_window_types:
                    alert =  self.generate_alert(topic_id , type_window1=target_window , type_window2=other_window)
                    if alert is not None:
                        yield alert


    def topic_pvalue_matrix(self , topic_id , trim = 0 ):

        topic_samples = self.samples[topic_id]
        nb_windows = len(self.types_window)
        pvalue_matrix = np.zeros((nb_windows , nb_windows))
        for i in range(nb_windows):
            a = topic_samples[self.types_window[i]]
            for j in range(i , nb_windows):
                b = topic_samples[self.types_window[j]]
                _ , pvalue = ttest_ind(a , b , trim=trim)
                pvalue_matrix[i][j] = pvalue
                pvalue_matrix[j][i] = pvalue
        return pvalue_matrix