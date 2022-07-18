import os
from typing import Dict
import json
import argparse
import pickle
from novelties_detection.Service.utils import initialize_calculator , createApp
from novelties_detection.Service.server_module import CollectThread ,NoveltiesDetectionThread
from novelties_detection.Experience.WindowClassification import WindowClassifierModel

parser = argparse.ArgumentParser(description="pass config_file with model , kwargs_calculator paths",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("config_path", default="config/default_config_service.json", help="paths of the model and kwargs calculator")
parser.add_argument("-l", "--length", default=20, type=int, help="Length of time series length cache")

args = parser.parse_args()
args = vars(args)
ROOT = os.getcwd()
config_path = args["config_path"]
LENGTH_CACHE = args["length"]
config_path = os.path.join(ROOT , config_path)
with open(config_path , 'r') as f:
    config = json.load(f)

MODEL_PATH = os.path.join(ROOT , config["model_path"])
KWARGS_CALCULATOR_PATH = os.path.join(ROOT , config["kwargs_calculator_path"])
KWARGS_MICRO_CALCULATOR_PATH = os.path.join(ROOT , config["kwargs_micro_calculator_path"])
PROCESSOR_PATH = os.path.join(ROOT , config["processor_path"])
LOOP_DELAY_PROCESS = config["loop_delay_process"] #minutes
LOOP_DELAY_COLLECT = config["loop_delay_collect"]
RSS_FEEDS_PATH = os.path.join(ROOT, config["rss_feeds_path"])
OUTPUT_PATH = config["output_path"]
HOST = config["host"]
PORT = config["port"]

if LOOP_DELAY_COLLECT > LOOP_DELAY_PROCESS:
    raise Exception("collect loop delay can't be superior to process loop delay ")


def load_stuff() -> Dict:
    """

    @return:
    """
    global LOOP_DELAY_PROCESS
    global LENGTH_CACHE
    with open(MODEL_PATH , "rb") as f:
        model : WindowClassifierModel = pickle.load(f)
    with open(KWARGS_CALCULATOR_PATH , "rb") as f:
        kwargs_calculator = pickle.load(f)
        kwargs_calculator["initialize_engine"]["memory_length"] = LENGTH_CACHE
    with open(KWARGS_MICRO_CALCULATOR_PATH , "rb") as f:
        kwargs_micro_calculator = pickle.load(f)
        kwargs_micro_calculator["initialize_engine"]["memory_length"] = LENGTH_CACHE
    with open(PROCESSOR_PATH , "rb") as f:
        processor = pickle.load(f)
    stuff : Dict = initialize_calculator(kwargs_calculator , n=1)
    micro_stuff : Dict = initialize_calculator(kwargs_micro_calculator)
    stuff.update({
        "classifier_models" : [model,model] , "loop_delay" : LOOP_DELAY_PROCESS , "processor" : processor,
        "micro_calculator" : micro_stuff["calculator"] , "micro_training_args" : micro_stuff["training_args"],

    })
    stuff["supervised_calculator"] = stuff["calculator"]
    del stuff["calculator"]
    del stuff["comparaison_args"]["back"]
    del stuff["comparaison_args"]["first_w"]
    del stuff["comparaison_args"]["last_w"]
    return stuff


def launch_app(stuff):
    injected_object_apis = [
        {"rss_feed_path": RSS_FEEDS_PATH},
        {"calculator": stuff["supervised_calculator"]},
        {"calculator": stuff["supervised_calculator"], "topics_finder": stuff["micro_calculator"]}
    ]
    app = createApp(injected_object_apis)
    app.run(HOST, port=PORT, debug=False)
    print("Running Server")


def startServer():
    '''
    Starts server
    :return:
    '''
    global RSS_FEEDS_PATH
    global OUTPUT_PATH
    global LOOP_DELAY_COLLECT
    stuff = load_stuff()
    extractor = CollectThread(RSS_FEEDS_PATH,OUTPUT_PATH,stuff["processor"],loop_delay=LOOP_DELAY_COLLECT)
    del stuff["processor"]
    detector = NoveltiesDetectionThread(**stuff)
    extractor.start()
    detector.start()
    launch_app(stuff)






if __name__ == '__main__':

    startServer()
    pass
