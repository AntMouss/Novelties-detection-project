import json

from novelties_detection.Experience.config_path import DATA_PATH, SEED_PATH, THEMATICS_PATH
from novelties_detection.Experience.data_utils import Thematic
from novelties_detection.Collection.data_processing import ProcessorText
from novelties_detection.Experience.Sequential_Module import LFIDFSequentialSimilarityCalculator , LDASequentialSimilarityCalculator , CoreXSequentialSimilarityCalculator , GuidedCoreXSequentialSimilarityCalculator , GuidedLDASequentialSimilarityCalculator
from novelties_detection.Collection.data_processing import exponentialThresholding , linearThresholding

#extremeum = [(1622900733.0  , Sat, 05 Jun 2021 15:45:33 GMT) , (1624642856.0 , Fri, 25 Jun 2021 19:40:56 GMT)]
NB_DAYS = 20
NB_HOURS = 1000
# 30/05/2021 14:01:40
START_DATE = 1622376100.0
LOOKBACK = 10
DELTA = 1
END_DATE = START_DATE + NB_HOURS * 3600
# load seed
with open(SEED_PATH, 'r') as f:
    SEED = json.load(f)

LABELS_IDX = list(SEED.keys())

NB_TOPICS = len(LABELS_IDX)

# load thematics
with open(THEMATICS_PATH, 'r') as f:
    THEMATICS = json.load(f)
    THEMATICS = [Thematic(**thematic) for thematic in THEMATICS]

# with open(MICRO_THEMATICS_PATH , "rb") as f:
#     MICRO_THEMATICS = pickle.load(f)


PROCESSOR = ProcessorText()

KWARGS = {
    #GuidedLDAModelKwargs,LFIDFModelKwargs,GuidedCoreXKwargs
    "kwargs_calculator_type": [LFIDFSequentialSimilarityCalculator ,
                               LDASequentialSimilarityCalculator ,
                               CoreXSequentialSimilarityCalculator ,
                               GuidedCoreXSequentialSimilarityCalculator ,
                               GuidedLDASequentialSimilarityCalculator],
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





