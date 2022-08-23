import os
import argparse
import pickle
from novelties_detection.Service.utils import createApp
from novelties_detection.Service.server_module import CollectThread ,NoveltiesDetectionThread
from config.config_server import (
    model_path,
    LOOP_DELAY_PROCESS,
    LOOP_DELAY_COLLECT,
    rss_feeds_path,
    output_path,
    HOST,
    PORT,
    macro_calculator_kwargs,
    micro_calculator_kwargs,
    macro_training_args,
    micro_training_args,
    macro_kwargs_results,
    PROCESSOR
)


DEFAULT_LENGTH_CACHE = 20

parser = argparse.ArgumentParser(description="pass config_file with model , kwargs_calculator paths",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--length", default=DEFAULT_LENGTH_CACHE, type=int, help="Length of time series length cache")

args = parser.parse_args()
args = vars(args)
ROOT = os.getcwd()
MODELS_PATH = os.path.join(ROOT, model_path)
RSS_FEEDS_PATH = os.path.join(ROOT, rss_feeds_path)
OUTPUT_PATH = output_path
LOOP_DELAY_PROCESS = LOOP_DELAY_PROCESS
LOOP_DELAY_COLLECT = LOOP_DELAY_COLLECT
HOST = HOST
PORT = PORT
PROCESSOR = PROCESSOR

if LOOP_DELAY_COLLECT > LOOP_DELAY_PROCESS:
    raise Exception("collect loop delay can't be superior to process loop delay ")

LENGTH_CACHE = args["length"]

macro_calculator_kwargs.memory_length = LENGTH_CACHE
macro_type = macro_calculator_kwargs.calculator_type
MACRO_CALCULATOR = macro_type(**macro_calculator_kwargs["kwargs"])
MACRO_TRAININGS_ARGS = macro_training_args
MACRO_RESULTS_ARGS = macro_kwargs_results

micro_calculator_kwargs.memory_length = LENGTH_CACHE
micro_type = micro_calculator_kwargs.calculator_type
MICRO_CALCULATOR = micro_type(**micro_calculator_kwargs["kwargs"])
MICRO_TRAININGS_ARGS = micro_training_args

with open(MODELS_PATH , "rb") as f:
    MODELS = pickle.load(f)


def start_Server():
    '''
    Starts server
    :return:
    '''
    global RSS_FEEDS_PATH
    global OUTPUT_PATH
    global LOOP_DELAY_COLLECT
    global LOOP_DELAY_PROCESS
    global PROCESSOR
    global MACRO_CALCULATOR
    global MICRO_CALCULATOR
    global MICRO_TRAININGS_ARGS
    global MACRO_TRAININGS_ARGS
    global MACRO_RESULTS_ARGS
    global MODELS

    extractor = CollectThread(
        rss_feed_source_path=RSS_FEEDS_PATH,
        output_path=OUTPUT_PATH ,
        processor=PROCESSOR,
        loop_delay=LOOP_DELAY_COLLECT)

    detector = NoveltiesDetectionThread(
        supervised_calculator=MACRO_CALCULATOR ,
        micro_calculator=MICRO_CALCULATOR ,
        training_args=MACRO_TRAININGS_ARGS,
        comparaison_args=MACRO_RESULTS_ARGS,
        micro_training_args = MICRO_TRAININGS_ARGS,
        loop_delay=LOOP_DELAY_PROCESS,
        classifier_models= MODELS
    )

    extractor.start()
    detector.start()
    injected_object_apis = [
        {"rss_feed_path": RSS_FEEDS_PATH},
        {"calculator": MACRO_CALCULATOR},
        {"calculator": MACRO_CALCULATOR, "topics_finder": MICRO_CALCULATOR}
    ]
    app = createApp(injected_object_apis)
    app.run(HOST, port=PORT, debug=False)
    print("Running Server")




if __name__ == '__main__':

    start_Server()
    pass
