from typing import Callable
# we assume that we import all the Sequential_Module to simplify settings customization
from novelties_detection.Experience import Sequential_Module
from novelties_detection.Experience.kwargs_utils import UpdateBadWordsKwargs
from novelties_detection.Collection.data_processing import absoluteThresholding ,  logarithmThresholding , linearThresholding , FrenchTextPreProcessor , MetaTextPreProcessor



# COLLECT SETTINGS
LOOP_DELAY_COLLECT : int = 5#minutes
COLLECT_RSS_IMAGES : bool = True
COLLECT_ARTICLE_IMAGES : bool = True
COLLECT_HTML_ARTICLE_PAGE : bool = True
PRINT_LOG : bool = True


#LABEL SETTINGS
LABELS_IDX = ["general" , "crime" , "politique" , "economy" , "justice"]



# PROCESS SETTINGS
LOOP_DELAY_PROCESS : int = 15#minutes
MEMORY_LENGTH : int = 10


# TEX PRE-PROCESSOR SETTINGS
LANG : str = "fr"
LEMMATIZE : bool = True
REMOVE_STOP_WORDS : bool = True
REMOVE_NUMBERS : bool = True
REMOVE_SMALL_WORDS : bool = True
PREPROCESSOR : MetaTextPreProcessor = FrenchTextPreProcessor(
    lemmatize=LEMMATIZE,
    remove_stop_words=REMOVE_STOP_WORDS,
    remove_numbers=REMOVE_NUMBERS,
    remove_small_words=REMOVE_SMALL_WORDS
)


# MACRO-CALCULATOR SETTINGS
MACRO_CALCULATOR_TYPE : type = Sequential_Module.TFIDFSequentialSimilarityCalculator
# macro_training_args = {
#     "passes" : 2 ,
#     "seed_strength" : 1000
# }
macro_training_args : dict = {}
macro_kwargs_results : dict = {
    "ntop" : 100,
    "remove_seed_words" : True,
    "back"  :  3
}



# MICRO-CALCULATOR SETTINGS
NB_MI_TOPICS = 7
MICRO_CALCULATOR_TYPE : type = Sequential_Module.LDASequentialSimilarityCalculatorFixed
micro_training_args : dict = {
    "passes" : 2
}



#BAD WORDS SETTINGS
# for remove words that no satisfing some frequency condition
fct_above : Callable = logarithmThresholding
fct_below : Callable = linearThresholding
kwargs_above : dict = {
    "limit" : 0.5
}
kwargs_below : dict = {
    "slop" : 0.001
}
bad_words_kwargs = UpdateBadWordsKwargs(
    thresholding_fct_above=fct_above ,
    thresholding_fct_below=fct_below ,
    kwargs_above=kwargs_above  ,
    kwargs_below=kwargs_below
)

