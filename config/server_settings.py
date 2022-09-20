from typing import Callable

from novelties_detection.Experience import Sequential_Module
from novelties_detection.Experience.kwargs_utils import UpdateBadWordsKwargs
from novelties_detection.Collection.data_processing import exponentialThresholding , linearThresholding , FrenchTextPreProcessor , MetaTextPreProcessor



#collect info
LOOP_DELAY_COLLECT : int = 5#minutes
COLLECT_RSS_IMAGES : bool = True
COLLECT_ARTICLE_IMAGES : bool = True
COLLECT_HTML_ARTICLE_PAGE : bool = True
PRINT_LOG : bool = True


# Process info
LOOP_DELAY_PROCESS : int = 15#minutes
MEMORY_LENGTH : int = 10


# text pre-processing info
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




#macro-calculator info
MACRO_CALCULATOR_TYPE : type = Sequential_Module.LFIDFSequentialSimilarityCalculator
# macro_training_args = {
#     "passes" : 2 ,
#     "overrate" : 1000
# }
macro_training_args : dict = {}
macro_kwargs_results : dict = {
    "ntop" : 100,
    "remove_seed_words" : True,
    "back"  :  3
}


#micro-calculator info
MICRO_CALCULATOR_TYPE : type = Sequential_Module.LDASequentialSimilarityCalculatorFixed
micro_training_args : dict = {"passes" : 2}
micro_kwargs_results : dict ={"ntop" : 100, "back" : 3}


#bad words info for remove words that no satisfing some frequency condition
fct_above : Callable = exponentialThresholding
fct_bellow : Callable = linearThresholding
kwargs_above : dict = {"limit" : 0.5 , "pente" : 100}
kwargs_bellow : dict = {"relative_value" : 0.001}
bad_words_kwargs = UpdateBadWordsKwargs(fct_above , fct_bellow , kwargs_above  , kwargs_bellow)

