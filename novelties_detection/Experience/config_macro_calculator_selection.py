from novelties_detection.Experience.kwargsGen import (KwargsAnalyse ,
                       KwargsResults ,
                       KwargsDataset ,
                       KwargsExperiences  ,
                        UpdateBadWordsKwargs,
                       LFIDFCalculatorKwargs  ,
                       GuidedCoreXCalculatorKwargs ,
                       GuidedLDACalculatorKwargs,
                        FullKwargs)
from novelties_detection.Experience.config_path import DATASET_PATH , MACRO_THEMATICS_PATH
from novelties_detection.Collection.data_processing import transformS , linearThresholding ,absoluteThresholding
from novelties_detection.Experience.data_utils import TimeLineArticlesDataset
import pickle
from novelties_detection.Experience.config_arguments import SEED , LABELS_IDX , NB_TOPICS

with open(MACRO_THEMATICS_PATH , "rb") as f:
    macro_thematics = pickle.load(f)
macro_thematics_dataset_id = macro_thematics["dataset_id"]
macro_thematics = macro_thematics["thematics"]

with open(DATASET_PATH , "rb") as f:
    res = pickle.load(f)
original_dataset : TimeLineArticlesDataset = res[1]
original_dataset_id = res[0]

if macro_thematics_dataset_id != original_dataset_id:
    raise Exception("the original dataset is not related to this macro thematics")

#fetch main information of the original dataset to instanciate new dataset with same attributs
# this solution isn't clean
PATH = original_dataset.path
START = original_dataset.start_date
END = original_dataset.end_date
DELTA = original_dataset.delta / 3600
TRANSFORM_FCT = transformS
LANG = "fr"

#STATIC VARIABLE
TRIM = 0.05
RISK = 0.05
NTOP = 100
LOOKBACK = 100
BACK = 3

NB_EXPERIENCES = 45
TIMELINE_SIZE = len(original_dataset)
THEMACTICS = macro_thematics
MIN_THEMATIC_SIZE = 1000
MIN_SIZE_EXPERIENCE = 3
MAX_SIZE_EXPERIENCE_RELATIVE = 0.2

THRESHOLDING_FCT_ABOVE = linearThresholding
THRESHOLDING_FCT_BELLOW = absoluteThresholding
ABOVE_KWARGS = {"relative_value" : 0.6}
BELLOW_KWARGS = {"absolute_value" : 2}

OVERRATE = 1000
ANCHOR_STRENGTH = 4


dataset = TimeLineArticlesDataset(PATH , START , END , LANG , LANG , LOOKBACK , transform_fct=TRANSFORM_FCT)

#DEFAULT KWARGS

kwargs_analyse = KwargsAnalyse(TRIM , RISK)
kwargs_results = KwargsResults(0 , 0 , NTOP , True ,BACK)
kwargs_dataset = KwargsDataset(START , END , PATH , LOOKBACK , DELTA)
kwargs_experiences = KwargsExperiences(
    NB_EXPERIENCES , TIMELINE_SIZE , THEMACTICS , MIN_THEMATIC_SIZE , MIN_SIZE_EXPERIENCE , MAX_SIZE_EXPERIENCE_RELATIVE)
kwargs_bad_words = UpdateBadWordsKwargs(THRESHOLDING_FCT_ABOVE , THRESHOLDING_FCT_BELLOW , ABOVE_KWARGS , BELLOW_KWARGS)

# DEFAULT ENGINE KWARGS
default_lfidf_kwargs_engine = LFIDFCalculatorKwargs(
    nb_topics=NB_TOPICS ,
    bad_words_args=kwargs_bad_words ,
    labels_idx=LABELS_IDX
)
default_guidedlda_kwargs_engine = GuidedLDACalculatorKwargs(
    nb_topics=NB_TOPICS ,
    bad_words_args=kwargs_bad_words ,
    labels_idx=LABELS_IDX ,
    seed=SEED,
    overrate=OVERRATE,
    passes = 2
)
default_guidedcorex_kwargs_engine = GuidedCoreXCalculatorKwargs(
    nb_topics=NB_TOPICS,
    bad_words_args=kwargs_bad_words,
    labels_idx=LABELS_IDX,
    seed=SEED,
    anchor_strength=ANCHOR_STRENGTH
)

STATIC_KWARGS_GENERATOR = [
    FullKwargs(kwargs_dataset , kwargs_experiences , default_lfidf_kwargs_engine , kwargs_results , kwargs_analyse),
    FullKwargs(kwargs_dataset , kwargs_experiences , default_guidedlda_kwargs_engine , kwargs_results , kwargs_analyse),
    FullKwargs(kwargs_dataset , kwargs_experiences , default_guidedcorex_kwargs_engine , kwargs_results , kwargs_analyse)
]