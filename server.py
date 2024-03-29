import os
import pickle
import logging
import json
from novelties_detection.Service.server_utils import createApp
from novelties_detection.Service.endpoints.apis_utils import ServiceException
from novelties_detection.Service.server_module import CollectThread ,NoveltiesDetectionThread
from novelties_detection.Experience.Sequential_Module import GuidedSequantialLangageSimilarityCalculator
from config.server_settings import (
    LOOP_DELAY_PROCESS,
    LOOP_DELAY_COLLECT,
    COLLECT_RSS_IMAGES,
    COLLECT_ARTICLE_IMAGES,
    COLLECT_HTML_ARTICLE_PAGE,
    PRINT_LOG,
    macro_training_args,
    micro_training_args,
    macro_kwargs_results,
    PREPROCESSOR,
    LANG,
    MACRO_CALCULATOR_TYPE,
    MICRO_CALCULATOR_TYPE,
    bad_words_kwargs,
    MEMORY_LENGTH,
    NB_MI_TOPICS,
    LABELS_IDX
)

# set level logs
logging.basicConfig(level=logging.INFO)

# deactivate gensim logs
gensim_logger = logging.getLogger('gensim')
gensim_logger.setLevel(logging.WARNING)

#server info
HOST = "0.0.0.0"
PORT = 5000
rss_feeds_path = "config/RSS_feeds.json"
MAX_LENGTH_CACHE = 30

ROOT = os.getcwd()
RSS_FEEDS_PATH = os.path.join(ROOT, rss_feeds_path)
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
LOOP_DELAY_PROCESS = LOOP_DELAY_PROCESS
LOOP_DELAY_COLLECT = LOOP_DELAY_COLLECT
PREPROCESSOR = PREPROCESSOR
LABELS_IDX = LABELS_IDX
LANG = LANG


if OUTPUT_PATH is None and (COLLECT_HTML_ARTICLE_PAGE or COLLECT_ARTICLE_IMAGES or COLLECT_RSS_IMAGES):
    raise ServiceException("if you want collect (write) html page or rss images or articles images you need to specify an output directory")
else:
    COLLECT_KWARGS = {
        "collectFullHtml": COLLECT_HTML_ARTICLE_PAGE,
        "collectRssImage": COLLECT_RSS_IMAGES,
        "collectArticleImages": COLLECT_ARTICLE_IMAGES,
        "print_log": PRINT_LOG
    }

if LOOP_DELAY_COLLECT > LOOP_DELAY_PROCESS:
    raise ServiceException("collect loop delay can't be superior to process loop delay ")

if PREPROCESSOR.lang != LANG:
    raise ServiceException("the LANG in server_settings.py is different that the preprocessor lang in server_settings.py ")

if MEMORY_LENGTH > MAX_LENGTH_CACHE:
    raise ServiceException("the Memory length exceed the require max length")

with open("config/seed.json" , "r") as f:
    seed = json.load(f)
with open("model/model.pck" , "rb") as f:
    MODELS = [pickle.load(f)] * len(seed)
labels_seed = list(seed.keys())

if issubclass(MACRO_CALCULATOR_TYPE , GuidedSequantialLangageSimilarityCalculator):
    MACRO_CALCULATOR = MACRO_CALCULATOR_TYPE(
        bad_words_args=bad_words_kwargs.__dict__,
        labels_idx=LABELS_IDX,
        memory_length=MEMORY_LENGTH,
        seed = seed
    )
else:
    MACRO_CALCULATOR = MACRO_CALCULATOR_TYPE(
        bad_words_args=bad_words_kwargs.__dict__,
        labels_idx=LABELS_IDX ,
        memory_length= MEMORY_LENGTH
    )

MACRO_TRAININGS_ARGS = macro_training_args
MACRO_RESULTS_ARGS = macro_kwargs_results


MICRO_CALCULATOR = MICRO_CALCULATOR_TYPE(
    nb_topics=NB_MI_TOPICS,
    bad_words_args=bad_words_kwargs.__dict__,
    memory_length= MEMORY_LENGTH
)
MICRO_TRAININGS_ARGS = micro_training_args


def start_Server():
    '''
    Starts server
    :return:
    '''
    global RSS_FEEDS_PATH
    global OUTPUT_PATH
    global LOOP_DELAY_COLLECT
    global LOOP_DELAY_PROCESS
    global PREPROCESSOR
    global MACRO_CALCULATOR
    global MICRO_CALCULATOR
    global MICRO_TRAININGS_ARGS
    global MACRO_TRAININGS_ARGS
    global MACRO_RESULTS_ARGS
    global MODELS
    global COLLECT_KWARGS
    global LABELS_IDX

    extractor = CollectThread(
        rss_feed_source_path=RSS_FEEDS_PATH,
        output_path=OUTPUT_PATH ,
        preprocessor=PREPROCESSOR,
        loop_delay=LOOP_DELAY_COLLECT,
        collect_kwargs=COLLECT_KWARGS
    )

    detector = NoveltiesDetectionThread(
        supervised_calculator=MACRO_CALCULATOR ,
        micro_calculator=MICRO_CALCULATOR ,
        training_args=MACRO_TRAININGS_ARGS,
        results_args=MACRO_RESULTS_ARGS,
        micro_training_args = MICRO_TRAININGS_ARGS,
        loop_delay=LOOP_DELAY_PROCESS,
        classifier_models= MODELS
    )

    extractor.start()
    detector.start()
    injected_object_apis = [
        {"rss_feed_path": RSS_FEEDS_PATH , "labels" : LABELS_IDX},
        {"calculator": MACRO_CALCULATOR},
        {"calculator": MACRO_CALCULATOR, "topics_finder": MICRO_CALCULATOR}
    ]
    app = createApp(injected_object_apis)
    app.run(HOST, port=PORT, debug=False)
    print("Server Running")



if __name__ == '__main__':

    start_Server()
