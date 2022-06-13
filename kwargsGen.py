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
        self.overrate = overrate


class GuidedCoreXKwargs(GuidedModelKwargs):
    def __init__(self, anchor_strength, seed, labels_idx,
                 nb_topics: int):
        super().__init__(seed, labels_idx, nb_topics)
        self.model_type = GuidedCoreXSequentialModeling
        self.anchor_strength = anchor_strength


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
                 min_thematic_size: int, min_size: int, max_size: int, cheat: bool, boost: int):
        self.boost = boost
        self.cheat = cheat
        self.max_size = max_size
        self.min_thematic_size = min_thematic_size
        self.min_size = min_size
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

KWARGS = {
    "kwargs_model_type": MetaModelKwargs.__subclasses__(),
    "nb_experiences": [32, 48, 64],
    "thematics": [thematics],
    "min_thematic_size": [0, 500, 1000, 4000],
    "min_size": [i for i in range(2, 10)],
    "max_size": [0.1, 0.2, 0.3],
    "cheat": [False],
    "boost": [0],
    "start": [1622376100],
    "end": [1622376100 + 25 * 24 * 3600],
    "path": ["/home/mouss/data/final_database_50000_100000_process_without_key.json"],
    "lookback": [i for i in range(5, 100, 5)],
    "delta": [1, 24],
    "processor": [processor],
    "nb_topics": [len(labels_idx)],
    "labels_idx": [labels_idx],
    "topic_id": [i for i in range(len(labels_idx))],
    "first_w": [0],
    "last_w": [0],
    "ntop": [30, 100],
    "fixeWindow": [False],
    "remove_seed_words": [True, False],
    "exclusive": [True, False],
    "back": [2, 8],
    "soft": [True, False],
    "random_state": [42],
    "overratte": [10 ** i for i in range(2, 7)],
    "anchor_strength": [i for i in range(3, 30)],
    "trim": [0, 1, 2]
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
    def __new__(cls):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("boost"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("cheat"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("max_size"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("min_thematic_size"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("min_size"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("thematics"))
        kwargs_dictionnary["timeline_size"] = math.ceil((KWARGS["end"] - KWARGS["start"])/KWARGS["delta"])
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
        return KwargsExperiences(**kwargs_dictionnary)


class KwargsResultsGenerator:
    def __new__(cls):
        kwargs_dictionnary = {}
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("remove_seed_words"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("fixeWindow"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("last_w"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("first_w"))
        kwargs_dictionnary.update(KwargsModelGenerator.choose_arg("topic_id"))
        return KwargsResults(**kwargs_dictionnary)


class FullKwargsGenerator:

    def __new__(cls):
        return {"experience" : KwargsExperiencesGenerator().__dict__ ,
                "initialize_dataset": KwargsDatasetGenerator().__dict__,
                "initialize_engine" : KwargsModelGenerator().__dict__ ,
                "generate_result": KwargsResultsGenerator().__dict__ }

class Generator:
    def __init__(self , n : int):
        self.n = n
    def __iter__(self):
        yield FullKwargsGenerator()