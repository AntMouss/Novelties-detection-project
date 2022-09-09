"""
this module isn't made to be used.
used to select best macro calculator with MACRO_THEMATICS data and associated window articles dataset that aren't available on this repo.
"""
import random
from collections import Counter
from itertools import repeat
from multiprocessing import Pool
import bisect
from typing import List , Union , Tuple

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from novelties_detection.Collection.data_processing import transformS, transformU
from novelties_detection.Experience.Sequential_Module import LDASequentialSimilarityCalculatorFixed, \
    NoSupervisedFixedSequantialLangageSimilarityCalculator, LFIDFSequentialSimilarityCalculator
from novelties_detection.Experience.kwargs_utils import FullKwargsForExperiences
from novelties_detection.Experience.ExperienceGen import SupervisedExperiencesProcessor , MetadataGenerationException
from novelties_detection.Experience.data_utils import ExperiencesResults, Alerte, TimeLineArticlesDataset, \
    MacroThematic, ExperiencesMetadata, ArticlesDataset, MicroThematic
import pickle
import logging
from threading import Lock
from novelties_detection.Experience.Exception_utils import  AnalyseException , ResultsGenerationException
from novelties_detection.Experience.utils import timer_func

l = Lock()

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s' , level=logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.propagate = False

NB_BEST_CALCULATORS = 3
NB_CALCULATORS = 15


class MicroCalculatorKwargsGenerator:

    unsupervised_kernels = [LDASequentialSimilarityCalculatorFixed]#CoreXSequentialSimilarityCalculator

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


class CalculatorInfo:

    def __init__(self, calculator_id, score : float, resultats : ExperiencesResults = None, full_kwargs : FullKwargsForExperiences = None):
        self.full_kwargs = full_kwargs
        self.resultats = resultats
        self.calculator_id = calculator_id
        self.score = score
    def __lt__(self, other):
        return self.score < other.score
    def __gt__(self, other):
        return self.score > other.score

class MicroCalculatorInfo(CalculatorInfo):
    def __init__(self, calculator_id, score : float , nb_clusters : int , kernel_type : str):
        super().__init__(calculator_id, score)
        self.kernel_type = kernel_type
        self.nb_clusters = nb_clusters




class MetaCalculatorSelector:
    best_calculators = []
    calculators_info = []
    def __init__(self, kwargs_calculator_generator):
        self.kwargs_calculator_generator = kwargs_calculator_generator

    @timer_func
    def process_selection(self, full_kwargs : FullKwargsForExperiences, max_to_save : int, path = None):
        pass

    def run(self, max_to_save : int, nb_workers : int = 1, path = None):
        if nb_workers == 1:
            for i , kwargs in enumerate(self.kwargs_calculator_generator):
                self.process_selection(kwargs, max_to_save=max_to_save, path=path)
                print(f"-------------"
                    f"calculator numero {i} process over {len(self.kwargs_calculator_generator)} from selector {id(self)}"
                      f"-------------")

        else:
            with Pool(nb_workers) as p:
                p.starmap(self.process_selection, zip(self.kwargs_calculator_generator, repeat(max_to_save), repeat(path)))
        return self.best_calculators

    def save(self,save_path : str ) :
        to_save = {
            "calculators_info" : self.calculators_info,
            "best_calculators" : self.best_calculators
        }
        with open(save_path , "wb") as f:
            f.write(pickle.dumps(to_save))

    def update_best_calculators(self, calculator_id  , max_to_save : int):
        bisect.insort(self.best_calculators, calculator_id)
        if len(self.best_calculators) > max_to_save:
            del self.best_calculators[0]



class MacroCalculatorSelector(MetaCalculatorSelector):

    def __init__(self, kwargs_calculator_generator , experiences_metadata_generator : Union[List[Tuple[ExperiencesMetadata , List[MacroThematic]]] , MetadataGenerationException]):
        super().__init__(kwargs_calculator_generator)
        self.experiences_metadata_generator = experiences_metadata_generator

    @staticmethod
    def compute_calculator_score(alerts : List[Alerte] , average=True):
        total_energy_distance = 0
        try:
            for alert in alerts:
                total_energy_distance += alert.mean_distance
            if average:
                return total_energy_distance / len(alerts)
            else:
                return total_energy_distance
        except ZeroDivisionError:
            return 0

    @timer_func
    def process_selection(self, full_kwargs : FullKwargsForExperiences, max_to_save : int, path = None):

        global l
        calculator_id = id(full_kwargs)
        energy_distance = 0
        kwargs_dataset = full_kwargs["dataset_args"]
        experiences_results = None
        try:

            dataset = TimeLineArticlesDataset(**kwargs_dataset)
            experience_generator = SupervisedExperiencesProcessor(self.experiences_metadata_generator, dataset)
            experience_generator.generate_results(**full_kwargs["results_args"])
            experiences_results = ExperiencesResults(experience_generator.experiences_res , experience_generator.info)
            alerts = SupervisedExperiencesProcessor.analyse_results(experiences_results, **full_kwargs["analyse_args"])
            energy_distance = self.compute_calculator_score(alerts)
            print(f"hypothesis mean distance equals to {energy_distance} for calculator {calculator_id}")
        except ResultsGenerationException:
            pass
        except AnalyseException:
            pass
        except Exception as e:
            logger.critical(f"Unknown Problem affect runtime : {e}")
            raise
        finally:
            calculator_info = CalculatorInfo(calculator_id, energy_distance , experiences_results , full_kwargs)
            self.update_best_calculators(calculator_id, max_to_save)
            self.calculators_info.append(calculator_info)
            l.acquire()
            if path is not None:
                self.save(path)
            l.release()
            del experience_generator



class RandomMacroCalculatorSelector(MacroCalculatorSelector):
    """
    Select macro calculator with random kwargs parameters and choose the best according to many criteria
    """

    def __init__(self, kwargs_calculator_generator , experiences_metadata_generator : Union[List[Tuple[ExperiencesMetadata , List[MacroThematic]]] , MetadataGenerationException]):
        super().__init__(kwargs_calculator_generator , experiences_metadata_generator)




class MicroCalculatorSelector(MetaCalculatorSelector):


    def __init__(self, micro_calculator_kwargs_generator: MicroCalculatorKwargsGenerator, micro_thematics: List[MicroThematic],
                 dataset: ArticlesDataset, ref_calculator_kwargs : dict ):

        super().__init__(micro_calculator_kwargs_generator)
        self.ref_calculator_kwargs = ref_calculator_kwargs
        self.dataset = dataset
        self.micro_thematics = micro_thematics
        self.best_micro_calculator_idx = []
        self.microth_window_idx = [ micro_thematic.window_idx_begin for micro_thematic in self.micro_thematics]
        self.res = {}


    def select_micro(self, micro_calculator : NoSupervisedFixedSequantialLangageSimilarityCalculator,
                     training_args : dict, kernel_type : str, max_to_save : int):


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
                ref_calculator = LFIDFSequentialSimilarityCalculator(labels_idx=labels_unique , **self.ref_calculator_kwargs)
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
            for micro_calculator , training_args , kernel_type in self.kwargs_calculator_generator:
                self.select_micro(micro_calculator , training_args , kernel_type , max_to_save=max_to_save )
        else:
            with Pool(nb_workers) as p:
                p.starmap(self.select_micro, zip(self.kwargs_calculator_generator, repeat(max_to_save), repeat(path)))
        if path is not None:
            self.save(path)
        return self.best_calculators