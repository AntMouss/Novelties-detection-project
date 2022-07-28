"""
this module isn't made to be used.
used to select best micro calculator with MICRO_THEMATICS data that isn't available on this repo.
"""
import copy
#from novelties_detection.Experience.config_arguments import MICRO_THEMATICS
from novelties_detection.Experience.data_utils import MicroThematic
from novelties_detection.Experience.Sequential_Module import (
MetaSequencialLangageSimilarityCalculator,
NoSupervisedSequantialLangageSimilarityCalculator,
LFIDFSequentialSimilarityCalculator,
LDASequentialSimilarityCalculator,
CoreXSequentialSimilarityCalculator
)
import numpy as np
from novelties_detection.Experience.kwargsGen import(
    LFIDFCalculatorKwargs ,
    UpdateBadWordsKwargs)
from novelties_detection.Collection.data_processing import transformU , transformS , linearThresholding ,absoluteThresholding


MAX_NB_CLUSTERS = 20
kwargs_above = {"relative_value" : 0.7}
kwargs_bellow = {"absolute_value" : 2}
labels_idx = ["target" , "None"]
kwargs_bad_words = UpdateBadWordsKwargs(linearThresholding , absoluteThresholding , kwargs_above , kwargs_bellow )
kwargs_ref_calculator = LFIDFCalculatorKwargs( nb_topics=2 , bad_words_args=kwargs_bad_words , labels_idx=labels_idx)
REF_CALCULATOR : LFIDFSequentialSimilarityCalculator(**kwargs_ref_calculator.__dict__)
kwargs_lda_micro_calculator = {
    "bad_words_args" : kwargs_bad_words,
}
kwargs_corex_micro_calculators= copy.deepcopy(kwargs_lda_micro_calculator)
kwargs_lda_micro_calculator["training_args"]["passes"] = 2
kwargs_micro_calculators = [kwargs_lda_micro_calculator , kwargs_corex_micro_calculators]


class MicroCalculatorGenerator:

    unsupervised_kernels = [LDASequentialSimilarityCalculator,CoreXSequentialSimilarityCalculator]

    def __init__(self, max_nb_clusters, kwargs_calculators):
        self.kwargs_calculators = kwargs_calculators
        self.max_nb_clusters = max_nb_clusters

    def __len__(self):

        return len(self.unsupervised_kernels) * self.max_nb_clusters

    def __iter__(self):
        return self

    def __next__(self) -> NoSupervisedSequantialLangageSimilarityCalculator :
        for i , kernel in enumerate(self.unsupervised_kernels):
            for nb_cluster in range(2, self.max_nb_clusters):
                self.kwargs_calculators[i]["nb_clusters"] = nb_cluster
                yield kernel(**self.kwargs_calculators[i])


class MicroCalculatorSelector:

    ref_calculator = REF_CALCULATOR

    def __init__(self , micro_calculator_generator : MicroCalculatorGenerator, micro_thematics):

        self.micro_thematics = micro_thematics
        self.micro_calculator_generator = micro_calculator_generator
        self.best_micro_calculator_idx = {"idx" : None , "purity" : 0}

    def select_micro_calculator(self):

        for idx ,  micro_calculator in self.micro_calculator_generator:
            purity_score = 0
            for micro_thematic , window_data in self.micro_thematics:
                window_data = MicroCalculatorSelector.add_labels(micro_thematic , window_data)
                supervised_window = transformS(window_data)
                unsupervised_window = transformU(window_data)
                micro_calculator.treat_Window(unsupervised_window)
                self.ref_calculator.treat_Window(supervised_window)
                micro_topics_words = micro_calculator.getTopWordsTopics(len(micro_calculator))
                # idx 1 is index to thematic topics
                ref_topic_words = self.ref_calculator.getTopWordsTopic(1 , len(self.ref_calculator))
                similarity_scores = []
                for micro_topic_words in micro_topics_words:
                    similarity_score , _ = MetaSequencialLangageSimilarityCalculator.compute_similarity(
                        micro_topic_words ,ref_topic_words)
                    similarity_scores.append(similarity_score)
                similarity_scores = np.array(similarity_scores)
                purity_score += np.max(similarity_scores)/np.sum(similarity_scores)
            if purity_score > self.best_micro_calculator_idx["purity"]:
                self.best_micro_calculator_idx = {"idx" : idx , "purity" : purity_score}
        return self.best_micro_calculator_idx

    @staticmethod
    def add_labels(micro_thematic : MicroThematic , window_data):
        for article in window_data:
            if article["id"] in micro_thematic.article_ids:
                article["label"] = "target"
            else:
                article["label"] = "None"
        return window_data


if __name__ == '__main__':

    micro_calculator_generator = MicroCalculatorGenerator(15 , kwargs_micro_calculators)
    # micro_calculator_selector = MicroCalculatorSelector(micro_calculator_generator , MICRO_THEMATICS)
    # best_micro_calculator = micro_calculator_selector.select_micro_calculator()
















