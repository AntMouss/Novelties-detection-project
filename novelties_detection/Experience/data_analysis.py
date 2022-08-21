""""
the purpose of this module is to see difference between different type of window.
In particular window that contain emulate topic change and normal window
The base hypothesis is: this two type of window have a different similarity score mean
"""

import functools
import numpy as np
import copy
from novelties_detection.Experience.data_utils import ExperiencesResults , Alerte , LabelisedSample
from scipy.stats import ttest_ind , normaltest  , energy_distance




def check_topic_id(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except IndexError:
            print("IndexError with topic_id...")
            print("note that if your results object come from no supervised calculator,"
                  " there is just one abstract topic with topic_id = 0")
    return wrapper


class SupervisedSampler:
    """
    format results data obtain during experience and transform it to being ready to use
    """

    def __init__(self , results : ExperiencesResults ):

        self.results = results.results
        self.info = results.info
        self.labels = self.info["labels"]

    def __len__(self):
        if self.info["mode"] == "u":
            return 1
        else:
            return self.info["nb_topics"]

    @property
    def samples(self) -> LabelisedSample:

        topic_samples = {
            "outside": [],
            "in": [],
            "out": [],
            "inside": []
        }
        samples = [copy.deepcopy(topic_samples) for _ in range(len(self))]
        for result in self.results:
            first_window_idx = result.metadata.begin_window_idx
            full_flat_ranges = []
            for thematic_ranges in result.metadata.ranges:
                full_flat_ranges += thematic_ranges
            full_flat_ranges = sorted(full_flat_ranges , key = lambda item : item[0])
            similarity = result.similarities_score
            for topic_id, difference_scores in enumerate(similarity):
                for window_id, difference_score in enumerate(difference_scores):
                    window_id = window_id + first_window_idx
                    key = SupervisedSampler.choose_key(window_id, full_flat_ranges)
                    samples[topic_id][key].append(difference_score)

        return LabelisedSample(self.labels, samples)


    @staticmethod
    def choose_key(idx_window , ranges):

        if idx_window < ranges[0][0]:
            return 'outside'
        elif idx_window > ranges[-1][1]:
            return 'outside'
        else:
            for entry , out in ranges:
                if idx_window == out:
                    return 'out'
                elif idx_window == entry:
                    return 'in'
                elif entry < idx_window < out:
                    return 'inside'
            return 'outside'


class Analyser:
    """
    analyse sample data to check hypothesis (normality and mean difference)
    """
    normality_risk = 0.05
    target_window_types = ['in', 'out']
    other_window_types = ['inside', 'outside']#'between' , 'after' , 'before'
    types_window = target_window_types + other_window_types

    def __init__(self, sample : LabelisedSample, risk = 0.05, trim = 0):
        """

        @param results:
        @param risk: alpha risk to check hypothesis availability
        @param trim: percent of aberrant sample we can remove for hypothesis checking
        """
        self.trim = trim
        self.risk = risk
        self.sample = sample
        self.nb_topics = len(self.sample)


    @check_topic_id
    def topic_pvalue_distance(self, topic_id : int, idx_window1 : int, idx_window2 : int, trim = 0):
        """
        return distance energy and pvalue of similarity score mean difference hypothesis between
        2 types of window for one specific topic.
        @param topic_id:
        @param idx_window1: target type window idx
        @param idx_window2:other type window idx
        @param trim:
        @return:
        """
        topic_samples = self.sample.samples_window[topic_id]
        target_samples = topic_samples[self.types_window[idx_window1]]
        other_samples = topic_samples[self.types_window[idx_window2]]
        try:
            _ , pvalue = ttest_ind(target_samples , other_samples , trim=trim)
            distance = energy_distance(target_samples,other_samples)
            mean_distance =  np.mean(other_samples) - np.mean(target_samples)
        except ValueError:
            return None , None , None
        return pvalue , distance , mean_distance



    @check_topic_id
    def test_hypothesis_topic_injection(self, topic_id : int, type_window1 : str, type_window2 : str
                                        , test_normality : bool = True):
        is_normal = self.test_hypothesis_normal_distribution(topic_id)
        if type_window1 not in self.types_window and type_window2 not in self.types_window:
            raise Exception(f"this windows type don't exist , here is "
                            f"the differents type:\n'{self.types_window}'")
        idx1 = self.types_window.index(type_window1)
        idx2 = self.types_window.index(type_window2)
        pvalue , en_distance , mean_distance = self.topic_pvalue_distance(topic_id, idx1, idx2, self.trim)
        if pvalue is None:
            return None
        elif pvalue < self.risk and (test_normality == False or (test_normality and is_normal) and mean_distance > 0) :
            return Alerte(topic_id , mean_distance , en_distance, risk=self.risk , windows=[type_window1 , type_window2]
                          , pvalue= pvalue , is_normal=is_normal)


    def multi_test_hypothesis_topic_injection(self, test_normality = True):

        for topic_id in range (self.nb_topics):
            for target_window in self.target_window_types:
                for other_window in self.other_window_types:
                    alert =  self.test_hypothesis_topic_injection(topic_id ,type_window1=target_window
                                    ,type_window2=other_window ,test_normality = test_normality)
                    if alert is not None:
                        yield alert

    @check_topic_id
    @functools.lru_cache(maxsize=2)
    def test_hypothesis_normal_distribution(self , topic_id):

        similarity_samples = []
        topic_samples = self.sample[topic_id]
        for window_type in self.other_window_types:
            similarity_samples += topic_samples[window_type]
        _ , pvalue = normaltest(similarity_samples)
        if pvalue >= self.normality_risk:
            print("Normality confirmed")
            return True
        else:
            print("Normality not confirmed")
            return False


    # def test_correlation_hypothesis_similarity_counter_articles(self):
    #
    #     for topic_id in range(self.nb_topics):
    #         similarity_topic = [result.similarities_score[topic_id] for result in
    #                             self.results.results]
    #         similarity_topic = list(itertools.chain.from_iterable(similarity_topic))
    #         count_topic = [[window_count[topic_id] for window_count in result.label_counter_ref] for result in
    #                               self.results.results]
    #         count_topic = list(itertools.chain.from_iterable(count_topic))
    #         corr, _ = pearsonr(similarity_topic, count_topic)
    #         print(f'Pearsons correlation for topic {topic_id}: %.3f' % corr)