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
from novelties_detection.Collection.data_processing import  absoluteThresholding , linearThresholding , exponentialThresholding
from novelties_detection.Experience.config_arguments import (
    Thematic , ProcessorText , THEMATICS , START_DATE , END_DATE , DATA_PATH , PROCESSOR , LABELS_IDX , SEED)

class UpdateBadWordsKwargs:
    def __init__(self,thresholding_fct_above: Callable,
                 thresholding_fct_bellow: Callable, kwargs_above: Dict, kwargs_bellow: Dict):
        self.kwargs_bellow = kwargs_bellow
        self.kwargs_above = kwargs_above
        self.thresholding_fct_bellow = thresholding_fct_bellow
        self.thresholding_fct_above = thresholding_fct_above

class MetaCalculatorKwargs:
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
                 min_thematic_size: int, min_size_exp: int, max_size_exp_rel: int):
        self.max_size_exp_rel = max_size_exp_rel
        self.min_thematic_size = min_thematic_size
        self.min_size_exp = min_size_exp
        self.thematics = thematics
        self.timeline_size = timeline_size
        self.nb_experiences = nb_experiences


class KwargsDataset:
    def __init__(self, start, end, path: str,
                 lookback: int, delta: int, processor: ProcessorText):
        self.processor = processor
        self.delta = delta
        self.lookback = lookback
        self.path = path
        self.end = end
        self.start = start


class KwargsResults:
    def __init__(self, first_w: int, last_w: int, ntop: int,
                  remove_seed_words: bool , back : int ):
        self.back = back
        self.remove_seed_words = remove_seed_words
        self.ntop = ntop
        self.last_w = last_w
        self.first_w = first_w

class KwargsNoSupervisedResults(KwargsResults):
    def __init__(self,reproduction_threshold : float , **kwargs):
        super().__init__(**kwargs)
        self.reproduction_threshold = reproduction_threshold


class KwargsAnalyse:
    def __init__(self , trim : int , risk = 0.05 ):
        self.trim = trim
        self.risk = risk



class MetaKwargsGenerator:

    @staticmethod
    def choose_arg(kwarg , key_name = None):
        if key_name is None:
            return {kwarg: random.choice(KWARGS[kwarg])}
        else:
            return {key_name: random.choice(KWARGS[kwarg])}

class KwargsBadWordsGenerator:
    def __new__(cls):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("nb_topics"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("thresholding_fct_above"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("thresholding_fct_bellow"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_kwargs_thresholding(
            [kwargs_dictionnary["thresholding_fct_above"], kwargs_dictionnary["thresholding_fct_bellow"]]))
        return UpdateBadWordsKwargs(**kwargs_dictionnary)

class KwargsModelGenerator(MetaKwargsGenerator):

    def __new__(cls , kwargs_calculator_type : Type[MetaCalculatorKwargs]):
        kwargs_dictionnary = {}
        kwargs_dictionnary["bad_words_args"] = KwargsBadWordsGenerator()
        if kwargs_calculator_type.__name__ == 'GuidedLDACalculatorKwargs':
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("overrate"))
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("passes"))
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("seed"))
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("labels_idx"))
        elif kwargs_calculator_type.__name__ == 'GuidedCoreXCalculatorKwargs':
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("anchor_strength"))
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("seed"))
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("labels_idx"))
        elif kwargs_calculator_type.__name__ == 'LFIDFCalculatorKwargs':
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("labels_idx"))
        elif kwargs_calculator_type.__name__ == 'CoreXCalculatorKwargs':
            pass
        elif kwargs_calculator_type.__name__ == 'LDACalculatorKwargs':
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("passes"))
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
                KwargsModelGenerator.choose_arg("absolute_value_above" , "absolute_value"))
        elif fcts[0] == linearThresholding:
            kwargs_thresholding["kwargs_above"].update(
                KwargsModelGenerator.choose_arg("relative_value_above", "relative_value"))
        elif fcts[0] == exponentialThresholding:
            kwargs_thresholding["kwargs_above"].update(
                KwargsModelGenerator.choose_arg("limit"))
            kwargs_thresholding["kwargs_above"].update(
                KwargsModelGenerator.choose_arg("pente"))
        if fcts[1] == absoluteThresholding:
            kwargs_thresholding["kwargs_bellow"].update(
                KwargsModelGenerator.choose_arg("absolute_value_bellow" , "absolute_value"))
        elif fcts[1] == linearThresholding:
            kwargs_thresholding["kwargs_bellow"].update(
                KwargsModelGenerator.choose_arg("relative_value_bellow", "relative_value"))
        return kwargs_thresholding




class KwargsExperiencesGenerator:
    def __new__(cls , timeline_size : int ):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("max_size_exp_rel"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("min_thematic_size"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("min_size_exp"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("thematics"))
        kwargs_dictionnary["timeline_size"] = timeline_size
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("nb_experiences"))
        return KwargsExperiences(**kwargs_dictionnary)


class KwargsDatasetGenerator:
    def __new__(cls):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("processor"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("delta"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("lookback"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("path"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("end"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("start"))
        return KwargsDataset(**kwargs_dictionnary)


class KwargsResultsGenerator:
    def __new__(cls, mode : str = "u"):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("remove_seed_words"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("last_w"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("first_w"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("ntop"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("back"))
        if mode == "s":
            return KwargsResults(**kwargs_dictionnary)
        else:
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("reproduction_threshold"))
            return KwargsNoSupervisedResults(**kwargs_dictionnary)


class KwargsAnalyseGenerator:
    def __new__(cls):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("risk"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("trim"))
        return KwargsAnalyse(**kwargs_dictionnary)



class FullKwargsGenerator:

    def __new__(cls , kwargs_calculator_type : Type[MetaCalculatorKwargs] = None):
        if kwargs_calculator_type is None:
            kwargs_calculator_type = random.choice(KWARGS["kwargs_calculator_type"])
        full_kwargs = {}
        full_kwargs["initialize_dataset"] = KwargsDatasetGenerator().__dict__
        end_date = full_kwargs["initialize_dataset"]["end"]
        start_date = full_kwargs["initialize_dataset"]["start"]
        delta = full_kwargs["initialize_dataset"]["delta"]
        timeline_size = math.ceil(( end_date- start_date) / (delta*3600))
        full_kwargs["experience"] = KwargsExperiencesGenerator(timeline_size).__dict__
        full_kwargs["initialize_engine"] = KwargsModelGenerator(kwargs_calculator_type).__dict__
        if issubclass(kwargs_calculator_type , SupervisedCalculatorKwargs):
            full_kwargs["generate_result"] = KwargsResultsGenerator(mode='s').__dict__
        else:
            full_kwargs["generate_result"] = KwargsResultsGenerator(mode='u').__dict__
        full_kwargs["analyse"] = KwargsAnalyseGenerator().__dict__
        return full_kwargs

class KwargsGenerator:
    def __init__(self , n : int):
        self.n = n

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield FullKwargsGenerator()


KWARGS = {
    #GuidedLDAModelKwargs,LFIDFModelKwargs,GuidedCoreXKwargs
    "kwargs_calculator_type": [GuidedLDACalculatorKwargs,
    GuidedCoreXCalculatorKwargs,
    LFIDFCalculatorKwargs,
    LDACalculatorKwargs,
    CoreXCalculatorKwargs],
    #32, 48, 64
    "nb_experiences": [32],
    "thematics": [THEMATICS],
    "min_thematic_size": [1000,2000],
    "min_size_exp": [i for i in range(2,4)],
    "max_size_exp_rel": [0.1, 0.2, 0.3],
    "start": [START_DATE],
    "end": [END_DATE],
    "path": [DATA_PATH],
    "lookback": [i for i in range(5, 100, 5)],
    "delta": [1],
    "processor": [PROCESSOR],
    "nb_topics": [len(LABELS_IDX)],
    "labels_idx": [LABELS_IDX],
    "topic_id": [i for i in range(len(LABELS_IDX))],
    "first_w": [0],
    "last_w": [0],
    "ntop": [ntop for ntop in range(50 , 101 , 10)],
    "seed" : [SEED],
    "remove_seed_words": [True],
    "exclusive": [True, False],
    "back": [2,3,4,5,6],
    "soft": [True, False],
    "random_state": [42],
    "overrate": [10 ** i for i in range(4, 7)],
    "anchor_strength": [i for i in range(3, 30)],
    "trim": [0, 0.05, 0.1],
    "risk" : [0.05],
    #absoluteThresholding , linearThresholding
    "thresholding_fct_above" : [exponentialThresholding , linearThresholding],
    #absoluteThresholding
    "thresholding_fct_bellow" : [linearThresholding],
    "absolute_value_above":[2000 , 10000 , 20000],
    "absolute_value_bellow" : [1, 3 , 10],
    "relative_value_above" : [0.7 , 0.5 , 0.25],
    "relative_value_bellow" : [0.05 , 0.01 , 0.001 , 0.0005 , 0.0001],
    "limit" : [0.7 , 0.5],
    "pente" : [100 , 500 , 1000],
    "passes" : [1 , 2 , 4 , 7],
   "reproduction_threshold"  : [0.1 , 0.2 , 0.3]
}
