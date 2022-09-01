
from novelties_detection.Experience import Sequential_Module
from novelties_detection.Experience.kwargsGen import KwargsResults , UpdateBadWordsKwargs
from novelties_detection.Collection.data_processing import exponentialThresholding , linearThresholding , FrenchTextPreProcessor


#server info
HOST = "127.0.0.1"
PORT = 5000
rss_feeds_path = "config/RSSfeeds_test.json"


#collect info
LOOP_DELAY_COLLECT = 5#minutes
COLLECT_RSS_IMAGES = False
COLLECT_ARTICLE_IMAGES = False
COLLECT_HTML_ARTICLE_PAGE = False
OUTPUT_PATH = None
PRINT_LOG = True


# Process info
LOOP_DELAY_PROCESS = 20#minutes
MEMORY_LENGTH = 10


# text pre-processing info
PREPROCESSOR = FrenchTextPreProcessor()
LANG = "fr"
LEMMATIZE = True
REMOVE_STOP_WORDS = True
REMOVE_NUMBERS = True
REMOVE_SMALL_WORDS = True



#macro-calculator info
MACRO_CALCULATOR_TYPE = Sequential_Module.LFIDFSequentialSimilarityCalculator
# macro_training_args = {
#     "passes" : 2 ,
#     "overrate" : 1000
# }
macro_training_args = {}
macro_kwargs_results = KwargsResults(
    ntop=100,
    remove_seed_words=True,
    back = 3
)


#micro-calculator info
MICRO_CALCULATOR_TYPE = Sequential_Module.LDASequentialSimilarityCalculatorFixed
micro_training_args = {"passes" : 2}
micro_kwargs_results = KwargsResults(
    ntop=100,
    back=3
)


#bad words info for remove words that no satisfing some frequency condition
fct_above = exponentialThresholding
fct_bellow = linearThresholding
kwargs_above = {"limit" : 0.5 , "pente" : 100}
kwargs_bellow = {"relative_value" : 0.001}
bad_words_kwargs = UpdateBadWordsKwargs(fct_above , fct_bellow , kwargs_above  , kwargs_bellow)

