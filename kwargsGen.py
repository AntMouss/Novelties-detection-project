from typing import Type, List
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
from data_processing import ProcessorText


class MetaModelKwargs:
    def __init__(self, nb_topics: int):
        self.nb_topics = nb_topics
        self.model_type = MetaSequencialLangageModeling
        self.training_args = {}


class SupervisedModelKwargs(MetaModelKwargs):
    def __init__(self, labels_idx, nb_topics: int):
        super().__init__(nb_topics)
        self.model_type = SupervisedSequantialLangagemodeling
        self.labels_idx = labels_idx


class GuidedModelKwargs(SupervisedModelKwargs):
    def __init__(self, seed, labels_idx, nb_topics: int):
        super().__init__(labels_idx, nb_topics)
        self.model_type = GuidedSequantialLangagemodeling
        self.seed = seed


class GuidedLDAModelKwargs(GuidedModelKwargs):
    def __init__(self, overrate, seed, labels_idx,
                 nb_topics: int):
        super().__init__(seed, labels_idx, nb_topics)
        self.model_type = GuidedLDASequentialModeling
        self.training_args["overrate"] = overrate


class GuidedCoreXKwargs(GuidedModelKwargs):
    def __init__(self, anchor_strength, seed, labels_idx,
                 nb_topics: int):
        super().__init__(seed, labels_idx, nb_topics)
        self.model_type = GuidedCoreXSequentialModeling
        self.training_args["anchor_strength"] = anchor_strength


class LFIDFModelKwargs(SupervisedModelKwargs):
    def __init__(self, labels_idx, nb_topics: int):
        super().__init__(labels_idx, nb_topics)
        self.model_type = LFIDFSequentialModeling


class LDAModelKwargs(MetaModelKwargs):
    def __init__(self, nb_topics: int):
        super().__init__(nb_topics)
        self.model_type = LDASequantialModeling


class CoreXModelKwargs(MetaModelKwargs):
    def __init__(self, nb_topics: int):
        super().__init__(nb_topics)
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
                 fixeWindow: bool, remove_seed_words: bool):
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
nb_jours = 1
start_date = 1622376100.0
end_date = start_date + nb_jours * 24 * 3600
lookback = 10
delta = 1
timeline_size = math.ceil((end_date - start_date) / delta)

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
    "min_thematic_size": [0, 500, 1000,2000],
    "min_size_exp": [i for i in range(2, 10)],
    "max_size_exp_rel": [0.1, 0.2, 0.3],
    "cheat": [False],
    "boost": [0],
    "start": [1622376100],
    "end": [1622376100 + nb_jours * 3 * 3600],
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
    "back": [2, 8],
    "soft": [True, False],
    "random_state": [42],
    "overrate": [10 ** i for i in range(2, 7)],
    "anchor_strength": [i for i in range(3, 30)],
    "trim": [0, 1, 2],
    "risk" : [0.05]

}

rel_kwargs = {
    "timeline_size"
}

class MetaKwargsGenerator:

    @staticmethod
    def choose_arg(kwarg):
        return {kwarg: random.choice(KWARGS[kwarg])}


class KwargsModelGenerator(MetaKwargsGenerator):

    def __new__(cls):
        kwargs_model_type = random.choice(KWARGS["kwargs_model_type"])
        kwargs_dictionnary = {}
        #kwargs_dictionnary["training_args"] = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("nb_topics"))
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
        timeline_size = math.ceil(( end_date- start_date) / delta)
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