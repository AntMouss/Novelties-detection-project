""""
use this server only for presentation and testing with dashboard
"""
import os
import json
import pickle
from novelties_detection.Service.server_utils import createApp
from novelties_detection.Experience.data_utils import TimeLineArticlesDataset
from novelties_detection.Collection.data_processing import transformU , transformS
from config.server_settings import (
    macro_kwargs_results,
    MICRO_CALCULATOR_TYPE,
    MEMORY_LENGTH,
    MACRO_CALCULATOR_TYPE,
    bad_words_kwargs,
    micro_training_args,
    macro_training_args
)
import pytest

#server info
HOST = "0.0.0.0"
PORT = 5000
rss_feeds_path = "novelties_detection/test/testing_data/RSSfeeds_test.json"

ROOT = os.getcwd()
RSS_FEEDS_PATH = os.path.join(ROOT, rss_feeds_path)


with open("novelties_detection/test/testing_data/test_seed.json", "r") as f:
    seed = json.load(f)
with open("novelties_detection/test/testing_data/test_model.pck", "rb") as f:
    MODELS = [pickle.load(f)] * len(seed)
labels_idx = list(seed.keys())


MACRO_CALCULATOR = MACRO_CALCULATOR_TYPE(
    bad_words_args=bad_words_kwargs.__dict__,
    labels_idx=labels_idx ,
    memory_length= MEMORY_LENGTH

)

MACRO_TRAININGS_ARGS = macro_training_args
MACRO_RESULTS_ARGS = macro_kwargs_results

MICRO_CALCULATOR = MICRO_CALCULATOR_TYPE(
    nb_topics=7,
    bad_words_args=bad_words_kwargs.__dict__,
    memory_length= MEMORY_LENGTH
)
MICRO_TRAININGS_ARGS = micro_training_args


DATA_PATH = "novelties_detection/test/testing_data/test_data_articles.json"
NB_WINDOW = 5
#2 juin 2021
START_DATE = 1622634860 + 10
END_DATE = START_DATE + NB_WINDOW * 3600 - 10

# we remove useless fields : "text" , "cleansed_text" , "content" , "summary" , "rss_media" , "images"
# from the original `testing_data/test_data_articles.json` keeping the project lighter(disk space).

training_supervised_dataset = TimeLineArticlesDataset(
    path=DATA_PATH,
    end=END_DATE,
    delta=1,
    start=START_DATE,
    lang="fr",
    lookback=10,
    transform_fct=transformS
)

training_unsupervised_dataset = TimeLineArticlesDataset(
    path=DATA_PATH,
    end=END_DATE,
    delta=1,
    lang="fr",
    start=START_DATE,
    lookback=10,
    transform_fct=transformU
)

@pytest.mark.skip(reason = "not testing function")
def run_static_Server():
    '''
    Starts server
    :return:
    '''
    global RSS_FEEDS_PATH
    global OUTPUT_PATH
    global LOOP_DELAY_COLLECT
    global RSS_FEEDS_PATH
    global MACRO_CALCULATOR
    global MICRO_CALCULATOR
    global MACRO_TRAININGS_ARGS
    global MICRO_TRAININGS_ARGS
    global HOST
    global PORT
    global training_supervised_dataset
    global training_unsupervised_dataset
    global labels_idx
    labels = labels_idx
    MACRO_CALCULATOR.add_windows(training_supervised_dataset , **MACRO_TRAININGS_ARGS)
    MICRO_CALCULATOR.add_windows(training_unsupervised_dataset , **MICRO_TRAININGS_ARGS)
    injected_object_apis = [
        {"rss_feed_path": RSS_FEEDS_PATH , "labels" : labels },
        {"calculator": MACRO_CALCULATOR},
        {"calculator": MACRO_CALCULATOR, "topics_finder": MICRO_CALCULATOR}
    ]
    app = createApp(injected_object_apis)
    app.run(HOST, port=PORT, debug=False)
    print("Server stops")

if __name__ == '__main__':
    run_static_Server()
