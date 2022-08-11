"""
this module isn't made to be used.
used to select best micro calculator with MICRO_THEMATICS data that isn't available on this repo.
"""
import os
import copy
import random
from collections import Counter
from itertools import repeat
from multiprocessing import Pool
from typing import List
from novelties_detection.Experience.data_utils import ArticlesDataset
from novelties_detection.Experience.config_calculator_selection import micro_thematics , original_dataset_hours
from novelties_detection.Experience.data_utils import MicroThematic
from novelties_detection.Experience.Sequential_Module import (
NoSupervisedSequantialLangageSimilarityCalculator,
LFIDFSequentialSimilarityCalculator,
LDASequentialSimilarityCalculator,
CoreXSequentialSimilarityCalculator
)
import numpy as np
from novelties_detection.Experience.kwargsGen import(
    UpdateBadWordsKwargs)
from novelties_detection.Collection.data_processing import transformU , transformS , linearThresholding ,absoluteThresholding
from sklearn.metrics import normalized_mutual_info_score
from macro_calculator_selection import MicroCalculatorInfo , MetaCalculatorSelector

MAX_NB_CLUSTERS = 20
kwargs_above = {"relative_value" : 0.7}
kwargs_bellow = {"absolute_value" : 2}
labels_idx = ["None" , "target0"]
kwargs_bad_words = UpdateBadWordsKwargs(linearThresholding , absoluteThresholding , kwargs_above , kwargs_bellow )
REF_CALCULATOR_kwargs = {"bad_words_args" : kwargs_bad_words.__dict__}
kwargs_lda_micro_calculator = {
    "bad_words_args" : kwargs_bad_words.__dict__,
    "training_args" : {}
}
kwargs_corex_micro_calculators= copy.deepcopy(kwargs_lda_micro_calculator)
kwargs_lda_micro_calculator["training_args"]["passes"] = 2
kwargs_micro_calculators = [kwargs_lda_micro_calculator , kwargs_corex_micro_calculators]


class MicroCalculatorGenerator:

    unsupervised_kernels = [LDASequentialSimilarityCalculator ]#CoreXSequentialSimilarityCalculator

    def __init__(self, max_nb_clusters : int, kwargs_calculators , seed : int = None):
        self.seed = seed
        self.kwargs_calculators = kwargs_calculators
        self.max_nb_clusters = max_nb_clusters

    def __len__(self):

        return len(self.unsupervised_kernels) * (self.max_nb_clusters - 1)

    def __iter__(self):
        if self.seed is not None:
            rValue = random.Random()
            rValue.seed(self.seed)
        else:
            rValue = None
        for  kwargs, kernel in zip(self.kwargs_calculators, self.unsupervised_kernels):
            training_args = kwargs["training_args"]
            del kwargs["training_args"]
            for nb_cluster in range(2, self.max_nb_clusters + 1):
                if rValue is not None:
                    random_state = rValue.randint(1 , 14340)
                    training_args["random_state"] = random_state
                    print(random_state)
                kwargs["nb_topics"] = nb_cluster
                yield kernel(**kwargs), training_args , kernel.__name__




class MicroCalculatorSelector(MetaCalculatorSelector):

    ref_calculator_kwargs = REF_CALCULATOR_kwargs

    def __init__(self, micro_calculator_generator: MicroCalculatorGenerator, micro_thematics: List[MicroThematic],
                 dataset: ArticlesDataset ):

        super().__init__()
        self.dataset = dataset
        self.micro_thematics = micro_thematics
        self.micro_calculator_generator = micro_calculator_generator
        self.best_micro_calculator_idx = []
        self.microth_window_idx = [ micro_thematic.window_idx_begin for micro_thematic in self.micro_thematics]
        self.res = {}


    def select_micro(self , micro_calculator : NoSupervisedSequantialLangageSimilarityCalculator ,
               training_args : dict, kernel_type : str , max_to_save : int):


        calculator_id = id(micro_calculator)
        purity_scores_with_none = []
        purity_scores_without_none = []
        nb_clusters = micro_calculator.nb_topics
        if kernel_type not in self.res.keys():
            self.res[kernel_type] = []
        for window_idx , ( _ , window_data) in enumerate(self.dataset):
            if window_idx > self.microth_window_idx[-1]:
                break
            if window_idx in self.microth_window_idx:
                micro_th_idxs = [thematic_idx for thematic_idx , w_idx  in enumerate(self.microth_window_idx) if w_idx==window_idx ]
                micro_thematics = [self.micro_thematics[micro_th_idx] for micro_th_idx in micro_th_idxs]
                window_data = MicroCalculatorSelector.change_labels(micro_thematics, window_data)
                supervised_window = transformS(window_data)
                unsupervised_window = transformU(window_data)
                labels = supervised_window[1]
                label_counter = dict(Counter(labels))
                label_counter = {k: v for k, v in sorted(label_counter.items(), key=lambda item: item[1] , reverse=True)}
                labels_unique = list(label_counter.keys())
                nb_topics = len(labels_unique)
                ref_calculator = LFIDFSequentialSimilarityCalculator(nb_topics , labels_idx=labels_unique , **self.ref_calculator_kwargs)
                micro_calculator.treat_Window(unsupervised_window , **training_args)
                ref_calculator.treat_Window(supervised_window)
                micro_topics_words = micro_calculator.getTopWordsTopics(len(micro_calculator) - 1 , ntop = 300, exclusive=True)
                # idx 1 is index to thematic topics
                ref_topics_words = ref_calculator.getTopWordsTopics(len(ref_calculator) - 1, ntop=300 , exclusive=True)
                purity_score_with_none , purity_score_without_none = MicroCalculatorSelector.compute_purity_score(ref_topics_words , micro_topics_words)
                purity_scores_with_none.append(purity_score_with_none)
                purity_scores_without_none.append(purity_score_without_none)
        average_purity_with_none = float(np.mean(purity_scores_with_none))
        average_purity_without_none = float(np.mean(purity_scores_without_none))
        self.res[kernel_type].append((average_purity_with_none , average_purity_without_none , nb_clusters ))
        micro_info = MicroCalculatorInfo(calculator_id , average_purity_without_none , nb_clusters , kernel_type)
        self.update_best_calculators(micro_info , max_to_save=max_to_save)

    @staticmethod
    def change_labels(micro_thematics : List[MicroThematic], window_data):
        for article in window_data:
            is_in_thematics_article = False
            for i, micro_thematic in enumerate(micro_thematics):
                if article["id"] in micro_thematic.article_ids:
                    label = f"target{i}"
                    article["label"] = [label]
                    is_in_thematics_article = True
            if is_in_thematics_article == False:
                article["label"] = ["None"]
        return window_data

    @staticmethod
    def compute_purity_score(ref_clusters_words : dict, clusters_words : List[dict]):
        labelized_clusters = MicroCalculatorSelector.labelize_clusters(ref_clusters_words, clusters_words)
        true_labels = []
        predict_labels = []
        for cluster in labelized_clusters : true_labels += cluster
        true_labels = np.array(true_labels)
        for cluster_id ,  cluster in enumerate(labelized_clusters) : predict_labels += [cluster_id] * len(cluster)
        predict_labels = np.array(predict_labels)
        purity_with_none_label= normalized_mutual_info_score(true_labels , predict_labels)
        none_label_mask = (true_labels != 0)
        true_labels_without_none = true_labels[none_label_mask]
        predict_labels_without_none = predict_labels[none_label_mask]
        purity_without_none_label = normalized_mutual_info_score(true_labels_without_none , predict_labels_without_none)
        return purity_with_none_label , purity_without_none_label

    @staticmethod
    def labelize_clusters(ref_clusters_words : dict, clusters_words : List[dict]):

        ref_clusters_words = [np.array(list(cluster_word.keys())) for cluster_word in ref_clusters_words]
        clusters_words = [np.array(list(cluster_words.keys())) for cluster_words in clusters_words]
        labelized_clusters = []
        for cluster_words in clusters_words:
            labelized_cluster = np.zeros_like(cluster_words , dtype= int)
            for cluster_id ,  ref_cluster_word in enumerate(ref_clusters_words):
                for word in ref_cluster_word:
                    bool_cluster = (cluster_words == word)
                    labelized_cluster[bool_cluster] = cluster_id
            labelized_cluster = labelized_cluster.astype(int)
            labelized_cluster = list(labelized_cluster)
            labelized_clusters.append(labelized_cluster)
        return labelized_clusters


    def run(self, max_to_save : int, nb_workers : int = 1, path = None):
        if nb_workers == 1:
            for micro_calculator , training_args , kernel_type in self.micro_calculator_generator:
                self.select_micro(micro_calculator , training_args , kernel_type , max_to_save=max_to_save )
        else:
            with Pool(nb_workers) as p:
                p.starmap(self.select_micro, zip(self.micro_calculator_generator, repeat(max_to_save), repeat(path)))
        if path is not None:
            self.save(path)
        return self.best_calculators




if __name__ == '__main__':
    root = "/home/mouss/PycharmProjects/novelties-detection-git/results"
    path = os.path.join(root , "LDA_micro_selection_res")
    micro_calculator_generator = MicroCalculatorGenerator(15 , kwargs_micro_calculators , 42 )
    micro_calculator_selector = MicroCalculatorSelector(micro_calculator_generator, micro_thematics, original_dataset_hours)
    best_micro_calculators = micro_calculator_selector.run(max_to_save=3 )


