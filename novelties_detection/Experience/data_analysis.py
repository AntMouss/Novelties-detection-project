import functools

import numpy as np
import copy
from novelties_detection.Experience.data_utils import ExperiencesResults , Alerte
from scipy.stats import ttest_ind , normaltest , pearsonr
import itertools


def check_topic_id(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IndexError:
            print("IndexError with topic_id...")
            print("note that if your results object come from no supervised calculator,"
                  " there is just one abstract topic with topic_id = 0")
    return wrapper


class Sampler:

    def __init__(self , results : ExperiencesResults ):

        self.results = results.results
        self.info = results.info

    def __len__(self):
        if self.info["mode"] == "u":
            return 1
        else:
            return self.info["nb_topics"]

    @property
    def samples(self):

        topic_samples = {
            "before": [],
            "after": [],
            "between": [],
            "in": [],
            "out": [],
            "inside": []
        }
        samples = [copy.deepcopy(topic_samples) for _ in range(len(self))]
        for result in self.results:
            similarity = result.similarity
            difference_matrix = abs(similarity['with'] - similarity['without'])
            for window_id, difference_scores in enumerate(difference_matrix):
                for topic_id , difference_score in enumerate(difference_scores):
                    key = Sampler.choose_key(window_id, result.metadata.ranges)
                    samples[topic_id][key].append(difference_score)

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
            return 'between'


class Analyser:

    def __init__(self , results : ExperiencesResults , risk = 0.05 , trim = 0):
        self.results = results
        self.trim = trim
        self.risk = risk
        self.samples = Sampler(self.results).samples
        self.nb_topics = len(self.samples)
        self.types_window = list(self.samples[0].keys())

    @property
    def pvalue_matrix(self):
        matrix = []
        for topic_id in range(self.nb_topics):
            matrix.append(self.topic_pvalue_matrix(topic_id, trim=self.trim))
        return matrix

    @check_topic_id
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


    @check_topic_id
    def test_hypothesis_topic_injection(self, topic_id : int, type_window1 : str, type_window2 : str
                                        , test_normality : bool = True):
        is_normal = self.test_hypothesis_normal_distribution(topic_id)
        if type_window1 not in self.types_window and type_window2 not in self.types_window:
            raise Exception(f"this windows type don't exist , here is "
                            f"the differents type:\n'{self.types_window}'")
        idx1 = self.types_window.index(type_window1)
        idx2 = self.types_window.index(type_window2)
        pvalue = self.pvalue_matrix[topic_id][idx1][idx2]
        if pvalue < self.risk:
            return Alerte(topic_id , risk=self.risk , windows=[type_window1 , type_window2]
                          , pvalue= pvalue , is_normal=is_normal)


    def multi_test_hypothesis_topic_injection(self, test_normality = True):

        target_window_types = ['in' , 'out']
        other_window_types = ['inside' , 'between' , 'after' , 'before']
        for topic_id in range (self.nb_topics):
            for target_window in target_window_types:
                for other_window in other_window_types:
                    alert =  self.test_hypothesis_topic_injection(topic_id ,type_window1=target_window
                                    ,type_window2=other_window ,test_normality = test_normality)
                    if alert is not None:
                        yield alert

    @check_topic_id
    @functools.lru_cache(maxsize=2)
    def test_hypothesis_normal_distribution(self , topic_id):
        # "without" key is linked to the result that we obtained with the reference model that isn't
        # contain topic injection
        #we assume that the distribution of similarity is invariant acoording to the label
        # so we flat the similarity["without"]
        similarity_samples = self.results.results[0].similarity["without"]
        similarity_samples = np.transpose(similarity_samples)
        similarity_samples = list(similarity_samples[topic_id])
        _ , pvalue = normaltest(similarity_samples)
        # we assume that we take the same risk for normality hypothesis and topic injection hypothesis
        if pvalue >= self.risk:
            print("Normality confirmed")
            return True
        else:
            print("Normality not confirmed")
            return False


    def test_correlation_hypothesis_similarity_counter_articles(self):

        for topic_id in range(self.nb_topics):
            similarity_topic = [result.similarity["without"][topic_id] for result in
                                  self.results.results]
            similarity_topic = list(itertools.chain.from_iterable(similarity_topic))
            count_topic = [[window_count[topic_id] for window_count in result.label_counter_ref] for result in
                                  self.results.results]
            count_topic = list(itertools.chain.from_iterable(count_topic))
            corr, _ = pearsonr(similarity_topic, count_topic)
            print(f'Pearsons correlation for topic {topic_id}: %.3f' % corr)