from typing import Type, List, Callable, Dict
import random
from Sequential_Module import (MetaSequencialLangageModeling,
                               GuidedSequantialLangagemodeling,
                               SupervisedSequantialLangagemodeling,
                               LFIDFSequentialModeling,
                               GuidedCoreXSequentialModeling,
                               GuidedLDASequentialModeling,
                               LDASequantialModeling,
                               NoSuperviedCoreXSequentialModeling,
                               )
import math
import json
from data_utils import Thematic
from data_processing import ProcessorText , absoluteThresholding , linearThresholding , exponentialThresholding


class MetaModelKwargs:
    def __init__(self, nb_topics: int , thresholding_fct_above: Callable,
                 thresholding_fct_bellow: Callable, kwargs_above: Dict, kwargs_bellow: Dict):
        self.thresholding_fct_above = thresholding_fct_above
        self.thresholding_fct_bellow = thresholding_fct_bellow
        self.kwargs_above = kwargs_above
        self.kwargs_bellow = kwargs_bellow
        self.nb_topics = nb_topics
        self.model_type = MetaSequencialLangageModeling
        self.training_args = {}


class SupervisedModelKwargs(MetaModelKwargs):
    def __init__(self, labels_idx, **kwargs):
        super().__init__(**kwargs)
        self.model_type = SupervisedSequantialLangagemodeling
        self.labels_idx = labels_idx


class GuidedModelKwargs(SupervisedModelKwargs):
    def __init__(self, seed , **kwargs):
        super().__init__(**kwargs)
        self.model_type = GuidedSequantialLangagemodeling
        self.seed = seed


class GuidedLDAModelKwargs(GuidedModelKwargs):
    def __init__(self, overrate , **kwargs):
        super().__init__(**kwargs)
        self.model_type = GuidedLDASequentialModeling
        self.training_args["overrate"] = overrate


class GuidedCoreXKwargs(GuidedModelKwargs):
    def __init__(self, anchor_strength , **kwargs):
        super().__init__(**kwargs)
        self.model_type = GuidedCoreXSequentialModeling
        self.training_args["anchor_strength"] = anchor_strength


class LFIDFModelKwargs(SupervisedModelKwargs):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.model_type = LFIDFSequentialModeling


class LDAModelKwargs(MetaModelKwargs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = LDASequantialModeling


class CoreXModelKwargs(MetaModelKwargs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = NoSuperviedCoreXSequentialModeling


class KwargsExperiences:
    def __init__(self, nb_experiences: int, timeline_size: int, thematics: List[Thematic],
                 min_thematic_size: int, min_size_exp: int, max_size_exp_rel: int, cheat: bool, boost: int):
        self.boost = boost
        self.cheat = cheat
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
    def __init__(self, topic_id: int, first_w: int, last_w: int, ntop: int,
                 fixeWindow: bool, remove_seed_words: bool , back : int):
        self.back = back
        self.remove_seed_words = remove_seed_words
        self.fixeWindow = fixeWindow
        self.ntop = ntop
        self.last_w = last_w
        self.first_w = first_w
        self.topic_id = topic_id

class KwargsAnalyse:
    def __init__(self , trim : int , risk = 0.05 ):
        self.trim = trim
        self.risk = risk



dataPath = '/home/mouss/data/final_database.json'
dataprocessedPath = '/home/mouss/data/final_database_50000_100000_process_without_key.json'
seedPath = '/home/mouss/data/mySeed.json'
all_experiences_file = '/home/mouss/data/myExperiences_with_random_state.json'
thematics_path = '/home/mouss/data/thematics.json'
nb_jours = 25
nb_hours = 15
start_date = 1622376100.0
lookback = 10
delta = 1

# load seed
with open(seedPath, 'r') as f:
    seed = json.load(f)

labels_idx = list(seed.keys())

# load thematics
with open(thematics_path, 'r') as f:
    thematics = json.load(f)
    thematics = [Thematic(**thematic) for thematic in thematics]

processor = ProcessorText()
#GuidedLDAModelKwargs,LFIDFModelKwargs,GuidedCoreXKwargs

KWARGS = {
    "kwargs_model_type": [GuidedLDAModelKwargs],
    "nb_experiences": [32, 48, 64],
    "thematics": [thematics],
    "min_thematic_size": [1000,2000],
    "min_size_exp": [i for i in range(2, 10)],
    "max_size_exp_rel": [0.1, 0.2, 0.3],
    "cheat": [False],
    "boost": [0],
    "start": [1622376100],
    "end": [1622376100 + nb_hours * 3600],
    "path": ["/home/mouss/data/final_database_50000_100000_process_without_key.json"],
    "lookback": [i for i in range(5, 100, 5)],
    "delta": [1],
    "processor": [processor],
    "nb_topics": [len(labels_idx)],
    "labels_idx": [labels_idx],
    "topic_id": [i for i in range(len(labels_idx))],
    "first_w": [0],
    "last_w": [0],
    "ntop": [ntop for ntop in range(50 , 101 , 10)],
    "fixeWindow": [False],
    "seed" : [seed],
    "remove_seed_words": [True],
    "exclusive": [True, False],
    "back": [2,3,4,5,6],
    "soft": [True, False],
    "random_state": [42],
    "overrate": [10 ** i for i in range(2, 7)],
    "anchor_strength": [i for i in range(3, 30)],
    "trim": [0, 1, 2],
    "risk" : [0.05],
    "thresholding_fct_above" : [absoluteThresholding , linearThresholding , exponentialThresholding],
    "thresholding_fct_bellow" : [absoluteThresholding , linearThresholding],
    "absolute_value_above":[2000 , 10000 , 20000],
    "absolute_value_bellow" : [1, 3 , 10],
    "relative_value_above" : [0.7 , 0.5 , 0.25],
    "relative_value_bellow" : [0.05 , 0.01 , 0.001 , 0.0005 , 0.0001],
    "limit" : [0.7 , 0.5 , 0.25],
    "pente" : [50 , 100 , 500 , 1000]

}

rel_kwargs = {
    "timeline_size"
}

class MetaKwargsGenerator:

    @staticmethod
    def choose_arg(kwarg , key_name = None):
        if key_name is None:
            return {kwarg: random.choice(KWARGS[kwarg])}
        else:
            return {key_name: random.choice(KWARGS[kwarg])}


class KwargsModelGenerator(MetaKwargsGenerator):

    def __new__(cls):
        kwargs_model_type = random.choice(KWARGS["kwargs_model_type"])
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("nb_topics"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("thresholding_fct_above" ))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("thresholding_fct_bellow" ))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_kwargs_thresholding(
            [kwargs_dictionnary["thresholding_fct_above"] , kwargs_dictionnary["thresholding_fct_bellow"]]))
        if kwargs_model_type.__name__ == 'GuidedLDAModelKwargs':
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("overrate"))
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("seed"))
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("labels_idx"))
        if kwargs_model_type.__name__ == 'GuidedCoreXKwargs':
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("anchor_strength"))
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("seed"))
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("labels_idx"))
        if kwargs_model_type.__name__ == 'LFIDFModelKwargs':
            kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("labels_idx"))
        if kwargs_model_type.__name__ == 'CoreXModelKwargs':
            pass
        if kwargs_model_type.__name__ == 'LDAModelKwargs':
            pass
        return kwargs_model_type(**kwargs_dictionnary)

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
    def __new__(cls , timeline_size):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("boost"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("cheat"))
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
    def __new__(cls):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("remove_seed_words"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("fixeWindow"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("last_w"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("first_w"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("topic_id"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("ntop"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("back"))
        return KwargsResults(**kwargs_dictionnary)


class KwargsAnalyseGenerator:
    def __new__(cls):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("risk"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("trim"))
        return KwargsAnalyse(**kwargs_dictionnary)



class FullKwargsGenerator:

    def __new__(cls):

        full_kwargs = {}
        full_kwargs["initialize_dataset"] = KwargsDatasetGenerator().__dict__
        end_date = full_kwargs["initialize_dataset"]["end"]
        start_date = full_kwargs["initialize_dataset"]["start"]
        delta = full_kwargs["initialize_dataset"]["delta"]
        timeline_size = math.ceil(( end_date- start_date) / (delta*3600))
        full_kwargs["experience"] = KwargsExperiencesGenerator(timeline_size).__dict__
        full_kwargs["initialize_engine"] = KwargsModelGenerator().__dict__
        full_kwargs["generate_result"] = KwargsResultsGenerator().__dict__
        full_kwargs["analyse"] = KwargsAnalyseGenerator().__dict__
        return full_kwargs

class KwargsGenerator:
    def __init__(self , n : int):
        self.n = n
    def __iter__(self):
        for i in range(self.n):
            yield FullKwargsGenerator()