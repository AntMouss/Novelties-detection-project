import json
import numpy as np
from novelties_detection.Experience.kwargsGen import (KwargsAnalyse ,
                                                      KwargsResults ,
                                                      KwargsDataset ,
                                                      KwargsExperiences  ,
                                                      UpdateBadWordsKwargs,
                                                      SupervisedCalculatorKwargs,
                                                        GuidedCalculatorKwargs,
                                                      FullKwargsForExperiences)
from novelties_detection.Experience.config_path import (
    DATASET_HOURS_PATH ,
    DATASET_DAYS_PATH ,
    MACRO_THEMATICS_HOURS_PATH ,
    MACRO_THEMATICS_DAYS_PATH ,
    MICRO_THEMATICS_PATH
)
from novelties_detection.Collection.data_processing import transformS ,absoluteThresholding , exponentialThresholding
from novelties_detection.Experience.data_utils import TimeLineArticlesDataset , MicroThematic , ExperiencesMetadata
import pickle
from novelties_detection.Experience.config_arguments import SEED , LABELS_IDX
from novelties_detection.Experience.Sequential_Module import (
    LFIDFSequentialSimilarityCalculator ,
    GuidedCoreXSequentialSimilarityCalculator ,
    GuidedLDASequentialSimilarityCalculator
)

def found_nan_window(dataset : TimeLineArticlesDataset , min_article):
    nan_window_idxs = []
    for date , window in dataset:
        if len(window) < min_article:
            nan_window_idxs.append(dataset.window_idx)
    return nan_window_idxs

def found_begin_window_idx(ranges , nan_window_idxs , gap = 2):
    tmp_first_window_ranges = []
    for ranges_thematic in ranges:
        for ranges in ranges_thematic:
            tmp_first_window_ranges.append(ranges[0])
    min_begin_window_idx = int(np.min(tmp_first_window_ranges) - gap)
    begin_window_idx = min_begin_window_idx
    while begin_window_idx in nan_window_idxs:
        begin_window_idx -= 1
    return begin_window_idx



with open(MACRO_THEMATICS_HOURS_PATH , "rb") as f:
    macro_thematics_hours = pickle.load(f)
macro_thematics_hours_dataset_id = macro_thematics_hours["dataset_id"]
macro_thematics_hours = macro_thematics_hours["thematics"]

with open(MACRO_THEMATICS_DAYS_PATH , "rb") as f:
    macro_thematics_days = pickle.load(f)
macro_thematics_days_dataset_id = macro_thematics_days["dataset_id"]
macro_thematics_days = macro_thematics_days["thematics"]

with open(MICRO_THEMATICS_PATH , "rb") as f:
    micro_thematics = json.load(f)
micro_thematics_dataset_id = micro_thematics["dataset_id"]
micro_thematics = micro_thematics["thematics"]
micro_thematics = [MicroThematic(**thematic) for thematic in micro_thematics]

with open(DATASET_HOURS_PATH , "rb") as f:
    res = pickle.load(f)
original_dataset_hours : TimeLineArticlesDataset = res[1]
original_dataset_hours_id = res[0]

with open(DATASET_DAYS_PATH , "rb") as f:
    res = pickle.load(f)
original_dataset_days : TimeLineArticlesDataset = res[1]
original_dataset_days_id = res[0]

if macro_thematics_hours_dataset_id != original_dataset_hours_id or micro_thematics_dataset_id != original_dataset_hours_id:
    raise Exception("the original dataset is not related to this macro thematics")

if macro_thematics_days_dataset_id != original_dataset_days_id:
    raise Exception("the original dataset is not related to this macro thematics")


#fetch main information of the original dataset to instanciate new dataset with same attributs
# this solution isn't clean
PATH = original_dataset_hours.path
START = original_dataset_hours.start_date
END = original_dataset_hours.end_date
DELTA = original_dataset_hours.delta / 3600
TRANSFORM_FCT = transformS
LANG = "fr"

#STATIC VARIABLE
TRIM = 0.05
RISK = 0.1
NTOP = 100
LOOKBACK = 50
BACK = 3

NB_EXPERIENCES = 10
TIMELINE_SIZE = 1000
THEMACTICS = macro_thematics_hours
MIN_THEMATIC_SIZE = 10
MIN_SIZE_EXPERIENCE = 3
MAX_SIZE_EXPERIENCE_RELATIVE = 0.8

THRESHOLDING_FCT_ABOVE = exponentialThresholding
THRESHOLDING_FCT_BELLOW = absoluteThresholding
ABOVE_KWARGS = {"limit" : 0.4 , "pente" : 100}
BELLOW_KWARGS = {"absolute_value" : 2}

OVERRATE = 1000
ANCHOR_STRENGTH = 4

NAN_WINDOW_IDX = found_nan_window( original_dataset_hours, 20)

#DEFAULT KWARGS

kwargs_analyse = KwargsAnalyse(TRIM , RISK)
kwargs_results = KwargsResults(NTOP , True ,BACK)
kwargs_dataset_hour = KwargsDataset(START, END, PATH, LOOKBACK, 1, transform_fct=transformS)
kwargs_dataset_days = KwargsDataset(START, END, PATH, LOOKBACK, 24, transform_fct=transformS)
kwargs_experiences = KwargsExperiences(
    NB_EXPERIENCES , TIMELINE_SIZE , THEMACTICS , MIN_THEMATIC_SIZE , MIN_SIZE_EXPERIENCE , MAX_SIZE_EXPERIENCE_RELATIVE)
kwargs_bad_words = UpdateBadWordsKwargs(THRESHOLDING_FCT_ABOVE , THRESHOLDING_FCT_BELLOW , ABOVE_KWARGS , BELLOW_KWARGS)

# DEFAULT ENGINE KWARGS
default_lfidf_kwargs_engine = SupervisedCalculatorKwargs(
    calculator_type= LFIDFSequentialSimilarityCalculator,
    bad_words_args=kwargs_bad_words ,
    labels_idx=LABELS_IDX
)
default_guidedlda_kwargs_engine = GuidedCalculatorKwargs(
    calculator_type= GuidedLDASequentialSimilarityCalculator,
    bad_words_args=kwargs_bad_words ,
    labels_idx=LABELS_IDX ,
    seed=SEED,
    training_args={
        "overrate" : OVERRATE,
        "passes" : 2
    }
)

default_guidedcorex_kwargs_engine = GuidedCalculatorKwargs(
    calculator_type= GuidedCoreXSequentialSimilarityCalculator,
    bad_words_args=kwargs_bad_words,
    labels_idx=LABELS_IDX,
    seed=SEED,
    training_args={
        "anchor_strength" : ANCHOR_STRENGTH
    }
)

STATIC_KWARGS_GENERATOR_HOURS = [
    FullKwargsForExperiences(kwargs_dataset_hour, kwargs_experiences, default_guidedcorex_kwargs_engine, kwargs_results,
                             kwargs_analyse),
    FullKwargsForExperiences(kwargs_dataset_hour, kwargs_experiences, default_guidedlda_kwargs_engine, kwargs_results,
                             kwargs_analyse),
    FullKwargsForExperiences(kwargs_dataset_hour, kwargs_experiences, default_lfidf_kwargs_engine, kwargs_results, kwargs_analyse)
]

STATIC_KWARGS_GENERATOR_DAYS = [
    FullKwargsForExperiences(kwargs_dataset_days, kwargs_experiences, default_lfidf_kwargs_engine, kwargs_results,
                             kwargs_analyse),
    FullKwargsForExperiences(kwargs_dataset_days, kwargs_experiences, default_guidedlda_kwargs_engine, kwargs_results, kwargs_analyse),
    FullKwargsForExperiences(kwargs_dataset_days, kwargs_experiences, default_guidedcorex_kwargs_engine, kwargs_results, kwargs_analyse)
]


HOUR_RANGES_1 = [[(15, 23) , (29 , 40) , (56, 92) ] ,
                 [],
                 [],
                 []]

HOUR_RANGES_2 = [[(105 , 114)  ] ,
                 [  (140 , 163) , (167,184)],
                 [],
                 []]

HOUR_RANGES_3 = [[(203 , 234) , (237 , 255) , (262 , 277)] ,
                 [(187 , 202)],
                 [],
                 []]

HOUR_RANGES_4 = [[(281 , 298) , (323 , 421)] ,
                 [],
                 [],
                 []]

HOUR_RANGES_5 = [[] ,
                 [(329 , 345) , (351 , 371) , (373 , 379) , (390 , 402)],
                 [],
                 []]

HOUR_RANGES_6 = [[(630 , 641)] ,
                 [],
                 [(586 , 610)],
                 []]

HOUR_RANGES_7 = [[] ,
                 [],
                 [(718 , 758) , (776 , 780)],
                 []]

HOUR_RANGES_8 = [[] ,
                 [],
                 [(807 , 839) , (852 , 901) ],
                 []]

HOUR_RANGES_9 = [[(424 , 501)] ,
                 [],
                 [],
                 []]

HOUR_RANGES_10 = [[(1046 , 1051) , (1055 , 1062)] ,
                 [],
                 [(1001 , 1019) , (1020 , 1045)],
                 [(1070 , 1089)]]

HOUR_RANGES_11 = [[] ,
                 [],
                 [(908 , 949) , (977 , 992) , (993 , 997) ],
                 []]

HOUR_RANGES_12 = [[] ,
                 [],
                 [(1312 , 1318)],
                 [(1213 , 1235) , (1257 , 1261)]]

HOUR_RANGES_13 = [[] ,
                 [ (1175 , 1187)],
                 [],
                 [(1090 , 1093)]]


HOUR_RANGES_LIST = [

    HOUR_RANGES_12,
    HOUR_RANGES_6,
    HOUR_RANGES_7,
    HOUR_RANGES_8,
    HOUR_RANGES_9,
    HOUR_RANGES_10,
    HOUR_RANGES_11,
    HOUR_RANGES_13,
    HOUR_RANGES_1,
    HOUR_RANGES_2,
    HOUR_RANGES_3,
    HOUR_RANGES_4,
    HOUR_RANGES_5
]


DAY_RANGES_1 =[
    [(43 , 45) , (47 , 49) , (62 , 64)],
    [ (4 , 5) , (8,9) , (10 , 13) , (15 , 17) , (19 , 20) , (22,24) , (26,27) , (29,30) , (37,38)],
    [(54,55) , (57,58)],
    []
]

DAY_RANGES_2 =[
    [(44 , 51) , (54 , 62) , (64 , 66)],
    [(33 , 37) , (38 , 42)],
    [(2,5)],
    [(7,8) , (10 , 11) , (14,16) , (18 , 20) , (23,24) , (27,29)]
]

DAY_RANGES_3 =[
    [],
    [],
    [(4,8)],
    [(9,13) , (16 , 17) , (18 , 21) , (23,28) ]
]


TEST_RANGES_HOURS_1 = [[(15, 23) , (29 , 40)] ,
                     [] ,
                     [] ,
                     []
                     ]

TEST_RANGES_HOURS_2 = [[(105 , 114)] ,
                     [(140 , 163)] ,
                     [] ,
                     []
                     ]

TEST_RANGES_DAYS_1 = [
    [],
    [],
    [(4,8)],
    [(9,13)]
]

TEST_RANGES_DAYS_2 = [
    [],
    [],
    [],
    [(16 , 17) , (18 , 21)]
]



EXPERIENCES_METADATA_GENERATOR_HOURS = [
    (ExperiencesMetadata("nimportequoihour" , hour_ranges , begin_window_idx=found_begin_window_idx(hour_ranges , NAN_WINDOW_IDX)) , macro_thematics_hours)
    for hour_ranges in HOUR_RANGES_LIST]



EXPERIENCES_METADATA_GENERATOR_DAYS = [
(ExperiencesMetadata("nimp1" , DAY_RANGES_1 , begin_window_idx=found_begin_window_idx(DAY_RANGES_1 , [])), macro_thematics_days),
(ExperiencesMetadata("nimp2" , DAY_RANGES_2 ,begin_window_idx=found_begin_window_idx(DAY_RANGES_2 , [])), macro_thematics_days),
(ExperiencesMetadata("nimp3" , DAY_RANGES_3 , begin_window_idx=found_begin_window_idx(DAY_RANGES_3 , [])), macro_thematics_days)
]


TEST_EXPERIENCES_METADATA_GENERATOR_HOURS = EXPERIENCES_METADATA_GENERATOR_HOURS[:2]


TEST_EXPERIENCES_METADATA_GENERATOR_DAYS = [
(ExperiencesMetadata("nimp2" , TEST_RANGES_DAYS_2 ,begin_window_idx=14), macro_thematics_days),
]
#(ExperiencesMetadata("nimp1" , TEST_RANGES_DAYS_1 , begin_window_idx= 2), macro_thematics_days),
