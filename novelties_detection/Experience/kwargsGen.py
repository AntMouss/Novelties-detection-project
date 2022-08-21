"""
this module isn't made to be used.
used to generate automaticely random kwargs that we need to make random experience
to select final macro calculator and micro calculator that we finaly use for the service.
the DATA_PATH pointed on the main window articles dataset that isn't available on this repo
"""
from typing import List, Callable, Dict, Type
import random
import math
from novelties_detection.Experience.Sequential_Module import (MetaSequencialLangageSimilarityCalculator,
                                                              GuidedSequantialLangageSimilarityCalculator,
                                                              SupervisedSequantialLangageSimilarityCalculator,
                                                              LFIDFSequentialSimilarityCalculator,
                                                              GuidedCoreXSequentialSimilarityCalculator,
                                                              GuidedLDASequentialSimilarityCalculator,
                                                              LDASequentialSimilarityCalculator,
                                                              CoreXSequentialSimilarityCalculator)
from novelties_detection.Collection.data_processing import  absoluteThresholding , linearThresholding , exponentialThresholding , transformS , transformU
from novelties_detection.Experience.config_arguments import Thematic , ProcessorText , KWARGS


class UpdateBadWordsKwargs:
    """
    data class that contain bad_words_args for update bad words with MetaSequentialCalculator instance
    """
    def __init__(self,thresholding_fct_above: Callable,
                 thresholding_fct_bellow: Callable, kwargs_above: Dict, kwargs_bellow: Dict):
        self.kwargs_bellow = kwargs_bellow
        self.kwargs_above = kwargs_above
        self.thresholding_fct_bellow = thresholding_fct_bellow
        self.thresholding_fct_above = thresholding_fct_above

class MetaCalculatorKwargs:
    """
    class that contain kwargs of MetaSequantialCalculator instance
    """
    def __init__(self, nb_topics: int , bad_words_args : UpdateBadWordsKwargs , training_args = None):
        self.bad_words_args = bad_words_args.__dict__
        self.nb_topics = nb_topics
        self.calculator_type = MetaSequencialLangageSimilarityCalculator
        if training_args is None:
            self.training_args = {}


class SupervisedCalculatorKwargs(MetaCalculatorKwargs):
    def __init__(self, labels_idx, **kwargs):
        super().__init__(**kwargs)
        self.calculator_type = SupervisedSequantialLangageSimilarityCalculator
        self.labels_idx = labels_idx


class GuidedCalculatorKwargs(SupervisedCalculatorKwargs):
    def __init__(self, seed , **kwargs):
        super().__init__(**kwargs)
        self.calculator_type = GuidedSequantialLangageSimilarityCalculator
        self.seed = seed

class LFIDFCalculatorKwargs(SupervisedCalculatorKwargs):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.calculator_type = LFIDFSequentialSimilarityCalculator


class LDACalculatorKwargs(MetaCalculatorKwargs):
    def __init__(self, passes, **kwargs):
        super().__init__(**kwargs)
        self.calculator_type = LDASequentialSimilarityCalculator
        self.training_args["passes"] = passes


class CoreXCalculatorKwargs(MetaCalculatorKwargs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calculator_type = CoreXSequentialSimilarityCalculator


class GuidedLDACalculatorKwargs(GuidedCalculatorKwargs , LDACalculatorKwargs):
    def __init__(self, overrate ,  **kwargs):
        super().__init__(**kwargs)
        self.calculator_type = GuidedLDASequentialSimilarityCalculator
        self.training_args["overrate"] = overrate


class GuidedCoreXCalculatorKwargs(GuidedCalculatorKwargs):
    def __init__(self, anchor_strength , **kwargs):
        super().__init__(**kwargs)
        self.calculator_type = GuidedCoreXSequentialSimilarityCalculator
        self.training_args["anchor_strength"] = anchor_strength



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
                 lookback: int, delta: int, processor: ProcessorText = None , transform_fct = None):
        self.transform_fct = transform_fct
        self.processor = processor
        self.delta = delta
        self.lookback = lookback
        self.path = path
        self.end = end
        self.start = start


class KwargsResults:
    def __init__(self, ntop: int,
                  remove_seed_words: bool , back : int ):
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
    def __init__(self , kwargs_dataset : KwargsDataset , kwargs_experiences : KwargsExperiences ,
                 kwargs_engine : MetaCalculatorKwargs , kwargs_result : KwargsResults ,
                 kwargs_analyse : KwargsAnalyse):
        self.kwargs_analyse = kwargs_analyse
        self.kwargs_result = kwargs_result
        self.kwargs_engine = kwargs_engine
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




class RandomMetaKwargsGenerator:

    @staticmethod
    def choose_arg(kwarg , key_name = None):
        if key_name is None:
            return {kwarg: random.choice(KWARGS[kwarg])}
        else:
            return {key_name: random.choice(KWARGS[kwarg])}

class RandomKwargsBadWordsGenerator:
    def __new__(cls):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("nb_topics"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("thresholding_fct_above"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("thresholding_fct_bellow"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_kwargs_thresholding(
            [kwargs_dictionnary["thresholding_fct_above"], kwargs_dictionnary["thresholding_fct_bellow"]]))
        return UpdateBadWordsKwargs(**kwargs_dictionnary)

class RandomKwargsCalculatorGenerator(RandomMetaKwargsGenerator):

    def __new__(cls, calculator_type : Type[MetaSequencialLangageSimilarityCalculator]):
        global kwargs_calculator_type
        kwargs_dictionnary = {}
        kwargs_dictionnary["bad_words_args"] = RandomKwargsBadWordsGenerator()
        if calculator_type.__name__ == 'GuidedLDASequentialSimilarityCalculator':
            kwargs_calculator_type = GuidedCalculatorKwargs
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("overrate"))
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("passes"))
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("seed"))
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("labels_idx"))
        elif calculator_type.__name__ == 'GuidedCoreXSequentialSimilarityCalculator':
            kwargs_calculator_type = GuidedCoreXCalculatorKwargs
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("anchor_strength"))
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("seed"))
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("labels_idx"))
        elif calculator_type.__name__ == 'LFIDFSequentialSimilarityCalculator':
            kwargs_calculator_type = LFIDFCalculatorKwargs
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("labels_idx"))
        elif calculator_type.__name__ == 'CoreXSequentialSimilarityCalculator':
            kwargs_calculator_type = CoreXCalculatorKwargs
            pass
        elif calculator_type.__name__ == 'LDASequentialSimilarityCalculator':
            kwargs_calculator_type = LDACalculatorKwargs
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("passes"))
            pass
        return kwargs_calculator_type(**kwargs_dictionnary)

    @staticmethod
    def choose_kwargs_thresholding(fcts : List[Callable]):
        kwargs_thresholding = {
            "kwargs_above" : {} ,
            "kwargs_bellow" : {}
        }
        #begin with kwargs_above geneneration
        if fcts[0] == absoluteThresholding:
            kwargs_thresholding["kwargs_above"].update(
                RandomKwargsCalculatorGenerator.choose_arg("absolute_value_above", "absolute_value"))
        elif fcts[0] == linearThresholding:
            kwargs_thresholding["kwargs_above"].update(
                RandomKwargsCalculatorGenerator.choose_arg("relative_value_above", "relative_value"))
        elif fcts[0] == exponentialThresholding:
            kwargs_thresholding["kwargs_above"].update(
                RandomKwargsCalculatorGenerator.choose_arg("limit"))
            kwargs_thresholding["kwargs_above"].update(
                RandomKwargsCalculatorGenerator.choose_arg("pente"))
        if fcts[1] == absoluteThresholding:
            kwargs_thresholding["kwargs_bellow"].update(
                RandomKwargsCalculatorGenerator.choose_arg("absolute_value_bellow", "absolute_value"))
        elif fcts[1] == linearThresholding:
            kwargs_thresholding["kwargs_bellow"].update(
                RandomKwargsCalculatorGenerator.choose_arg("relative_value_bellow", "relative_value"))
        return kwargs_thresholding




class RandomKwargsExperiencesGenerator:
    def __new__(cls , timeline_size : int ):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("max_size_exp_rel"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("min_thematic_size"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("min_size_exp"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("thematics"))
        kwargs_dictionnary["timeline_size"] = timeline_size
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("nb_experiences"))
        return KwargsExperiences(**kwargs_dictionnary)


class RandomKwargsDatasetGenerator:
    def __new__(cls , mode : str = 's'):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("processor"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("delta"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("lookback"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("path"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("end"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("start"))
        if mode == 's' :
            kwargs_dictionnary["transform_fct"] = transformS
        else:
            kwargs_dictionnary["transform_fct"] = transformU

        return KwargsDataset(**kwargs_dictionnary)


class RandomKwargsResultsGenerator:
    def __new__(cls, mode : str = "u"):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("remove_seed_words"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("ntop"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("back"))
        if mode == "s":
            return KwargsResults(**kwargs_dictionnary)
        else:
            kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("reproduction_threshold"))
            return KwargsNoSupervisedResults(**kwargs_dictionnary)


class RandomKwargsAnalyseGenerator:
    def __new__(cls):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("risk"))
        kwargs_dictionnary.update(RandomKwargsCalculatorGenerator.choose_arg("trim"))
        return KwargsAnalyse(**kwargs_dictionnary)



class RandomFullProcessKwargsGenerator:

    def __new__(cls , kwargs_calculator_type : Type[MetaCalculatorKwargs] = None):
        if kwargs_calculator_type is None:
            kwargs_calculator_type = random.choice(KWARGS["kwargs_calculator_type"])
        kwargs_engine = RandomKwargsCalculatorGenerator(kwargs_calculator_type)
        if issubclass(kwargs_calculator_type , SupervisedCalculatorKwargs):
            kwargs_result = RandomKwargsResultsGenerator(mode='s')
        else:
            kwargs_result = RandomKwargsResultsGenerator(mode='u')
        kwarg_analyse = RandomKwargsAnalyseGenerator()
        kwargs_dataset = RandomKwargsDatasetGenerator()
        end_date = kwargs_dataset.end
        start_date = kwargs_dataset.start
        delta = kwargs_dataset.delta
        timeline_size = math.ceil((end_date - start_date) / (delta * 3600))
        kwargs_experience = RandomKwargsExperiencesGenerator(timeline_size)
        return FullKwargs(kwargs_dataset , kwargs_experience , kwargs_engine , kwargs_result , kwarg_analyse)

class RandomKwargsGenerator:
    def __init__(self , n : int):
        self.n = n

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield RandomFullProcessKwargsGenerator()

