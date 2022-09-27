"""
this module isn't made to be used.
used to generate automaticely random kwargs that we need to make random experience
to select final macro calculator and micro calculator that we finaly use for the service.
the DATA_PATH pointed on the main window articles dataset that isn't available on this repo
"""
import copy
from typing import List, Callable, Dict, Type
from novelties_detection.Experience.Sequential_Module import (
    NoSupervisedSequantialLangageSimilarityCalculator,
    SupervisedSequantialLangageSimilarityCalculator,
    MetaSequencialLangageSimilarityCalculator
)
from novelties_detection.Collection.data_processing import FrenchTextPreProcessor
from novelties_detection.Experience.data_utils import Thematic


class UpdateBadWordsKwargs:
    """
    data class that contain bad_words_args for update bad words with MetaSequentialCalculator instance
    """
    def __init__(self, thresholding_fct_above: Callable,
                 thresholding_fct_below: Callable, kwargs_above: Dict, kwargs_below: Dict):
        self.kwargs_below = kwargs_below
        self.kwargs_above = kwargs_above
        self.thresholding_fct_below = thresholding_fct_below
        self.thresholding_fct_above = thresholding_fct_above

class CalculatorKwargs:
    """
       class that contain kwargs of MetaSequantialCalculator instance
       """

    def __init__(self, calculator_type: Type[MetaSequencialLangageSimilarityCalculator],
                 bad_words_args: UpdateBadWordsKwargs, memory_length: int = None, training_args: dict = None):
        self.memory_length = memory_length
        self.bad_words_args = bad_words_args.__dict__
        self.calculator_type = calculator_type
        if training_args is None:
            self.training_args = {}
        else:
            self.training_args = training_args

    def __getitem__(self, item):

        if item == "calculator_kwargs":
            tmp = copy.deepcopy(self.__dict__)
            del tmp["calculator_type"]
            del tmp["training_args"]
        elif item == "training_kwargs":
            tmp = self.training_args
        else:
            raise KeyError(f"{item}")
        keys_to_delete = []
        for key, item in tmp.items():
            if item is None:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del tmp[key]
        return tmp

class SupervisedCalculatorKwargs(CalculatorKwargs):
    """
    class that contain kwargs of MetaSequantialCalculator instance
    """
    def __init__(self, calculator_type: Type[SupervisedSequantialLangageSimilarityCalculator], labels_idx: List,
                 bad_words_args: UpdateBadWordsKwargs, memory_length: int = None,
                 training_args: dict = None):
        super().__init__(calculator_type, bad_words_args, memory_length, training_args)
        self.memory_length = memory_length
        self.labels_idx = labels_idx
        self.bad_words_args = bad_words_args.__dict__
        self.calculator_type = calculator_type
        if training_args is None:
            self.training_args = {}


class GuidedCalculatorKwargs(SupervisedCalculatorKwargs):
    def __init__(self, calculator_type: Type[SupervisedSequantialLangageSimilarityCalculator], labels_idx: List,
                 bad_words_args: UpdateBadWordsKwargs , seed : dict , memory_length: int = None,
                 training_args: dict = None):
        super().__init__(calculator_type, labels_idx, bad_words_args , memory_length ,training_args)
        self.seed = seed


class NoSupervisedCalculatorKwargs(CalculatorKwargs):
    def __init__(self, calculator_type: Type[NoSupervisedSequantialLangageSimilarityCalculator],
                 bad_words_args: UpdateBadWordsKwargs , nb_topics : int = None, memory_length: int = None, training_args: dict = None):
        super().__init__(calculator_type, bad_words_args , memory_length , training_args)
        self.nb_topics = nb_topics



class KwargsExperiences:
    def __init__(self, nb_experiences: int, timeline_size: int, thematics: List[Thematic],
                 min_thematic_size: int, min_size_exp: int, max_size_exp_rel: float):
        self.max_size_exp_rel = max_size_exp_rel
        self.min_thematic_size = min_thematic_size
        self.min_size_exp = min_size_exp
        self.thematics = thematics
        self.timeline_size = timeline_size
        self.nb_experiences = nb_experiences


class KwargsDataset:
    def __init__(self, start, end, path: str,
                 lookback: int, delta: int, processor: FrenchTextPreProcessor = None, transform_fct = None):
        self.transform_fct = transform_fct
        self.processor = processor
        self.delta = delta
        self.lookback = lookback
        self.path = path
        self.end = end
        self.start = start


class KwargsResults:
    def __init__(self, ntop: int,
                  remove_seed_words: bool = False , back : int  = 1):
        self.back = back
        self.remove_seed_words = remove_seed_words
        self.ntop = ntop

class KwargsNoSupervisedResults(KwargsResults):
    def __init__(self,reproduction_threshold : float , **kwargs):
        super().__init__(**kwargs)
        self.reproduction_threshold = reproduction_threshold


class KwargsAnalyse:
    def __init__(self , trim : float , risk = 0.05 ):
        self.trim = trim
        self.risk = risk



class FullKwargs:
    def __init__(self, kwargs_engine : SupervisedCalculatorKwargs, kwargs_result : KwargsResults):
        self.kwargs_result = kwargs_result
        self.kwargs_engine = kwargs_engine

    def __getitem__(self, item):

        kwargs = {}
        if item == 'results_args':
            kwargs.update(self.kwargs_result.__dict__)
        elif item == "calculator_args":
            kwargs.update(self.kwargs_engine.__dict__)
        else:
            raise KeyError(f"{item}")
        return kwargs


class FullKwargsForExperiences(FullKwargs):
    def __init__(self, kwargs_dataset: KwargsDataset, kwargs_experiences: KwargsExperiences,
                 kwargs_engine: SupervisedCalculatorKwargs, kwargs_result: KwargsResults, kwargs_analyse: KwargsAnalyse):
        super().__init__(kwargs_engine, kwargs_result)
        self.kwargs_analyse = kwargs_analyse
        self.kwargs_experiences = kwargs_experiences
        self.kwargs_dataset = kwargs_dataset

    def __getitem__(self, item):

        kwargs = {}
        if item == 'results_args':
            kwargs.update(self.kwargs_result.__dict__)
            kwargs.update({"calculator_args" : self["calculator_args"]})
        elif item == "calculator_args":
            kwargs.update(self.kwargs_engine.__dict__)
            kwargs.update({
                "dataset_args" : self["dataset_args"]
            })
        elif item == "dataset_args":
            kwargs.update(self.kwargs_dataset.__dict__)
        elif item == "experiences_args" :
            kwargs.update(self.kwargs_experiences.__dict__)
        elif item == "analyse_args":
            kwargs.update(self.kwargs_analyse.__dict__)
        else:
            raise KeyError(f"{item}")
        return kwargs
