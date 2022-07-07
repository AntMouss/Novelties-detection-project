import os.path
from typing import List, Dict
import json
from threading import Thread , Lock
import schedule
from Collection.RSSCollector import RSSCollect
from Experience.Sequential_Module import MetaSequencialLangageSimilarityCalculator , SupervisedSequantialLangageSimilarityCalculator
from Experience.WindowClassification import WindowClassifierModel
from Collection.data_processing import transformS
import argparse
import pickle
from flask import Blueprint, Flask
from flask_restx import Api
from Server.apis.rss_feed_api.namespaces import namesp

parser = argparse.ArgumentParser(description="pass config_file with model , kwargs_calculator paths",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("config_path", help="paths of the model and kwargs calculator")
parser.add_argument("root_path" , help="root path of the project")
args = parser.parse_args()
args = vars(args)
config_path = args["config_path"]
ROOT = args["root_path"]
config_path = os.path.join(ROOT , config_path)
with open(config_path , 'r') as f:
    config = json.load(f)

MODEL_PATH = os.path.join(ROOT , config["model_path"])
KWARGS_CALCULATOR_PATH = os.path.join(ROOT , config["kwargs_calculator_path"])
PROCESSOR_PATH = os.path.join(ROOT , config["processor_path"])
LOOP_DELAY_PROCESS = config["loop_delay_process"] #minutes
LOOP_DELAY_COLLECT = config["loop_delay_collect"]
RSS_FEEDS_PATH = os.path.join(ROOT, config["rss_feeds_path"])
OUTPUT_PATH = config["output_path"]
HOST = config["host"]
PORT = config["port"]


WINDOW_DATA = []
READY_TO_CONSUME = False
COLLECT_LOCKER = Lock()
PROCESS_LOCKER = Lock()


def initialize_calculator(kwargs_calculator):
    supervised_calculator_type = kwargs_calculator['initialize_engine']['model_type']
    training_args = kwargs_calculator['initialize_engine']['training_args']
    comparaison_args = kwargs_calculator['generate_result']
    del kwargs_calculator['initialize_engine']['calculator_type']
    del kwargs_calculator['initialize_engine']['training_args']
    sequential_model = supervised_calculator_type
    supervised_calculator: MetaSequencialLangageSimilarityCalculator = sequential_model(
        **kwargs_calculator['initialize_engine'])
    return {
        "supervised_calculator" : supervised_calculator ,
        "comparaison_args" : comparaison_args ,
        "training_args"  : training_args
    }


def load_stuff() -> Dict:
    """

    @return:
    """
    global model
    global LOOP_DELAY_PROCESS
    with open(MODEL_PATH , "rb") as f:
        model = pickle.load(f)
    with open(KWARGS_CALCULATOR_PATH , "rb") as f:
        kwargs_calculator_path = pickle.load(f)
    with open(PROCESSOR_PATH , "rb") as f:
        processor = pickle.load(f)
    stuff = initialize_calculator(kwargs_calculator_path)
    stuff.update({"classifier_models" : model , "loop_delay" : LOOP_DELAY_PROCESS , "processor" : processor})
    return stuff


class CollectThread(Thread):
    """
    Service to periodically collect information from the selected sources
    """
    def __init__(self,rss_feed_config_file,output_path ,processor, delta):
        Thread.__init__(self)
        self.delta = delta
        self.rss_feed_config=rss_feed_config_file
        self.output_path=output_path
        self.rssCollect=RSSCollect(self.rss_feed_config, self.output_path , processor=processor)
        self.loop_delay , self.nb_loop = self._init_loop_delay()
        self.count = 0

    def _init_loop_delay(self):
        max_loop_delay = 30
        min_loop_delay = 5
        for loop_delay in reversed(range(max_loop_delay , min_loop_delay )):
            if self.delta % loop_delay == 0:
                return loop_delay , self.delta//loop_delay


    def update_window_data(self):
        global WINDOW_DATA
        global READY_TO_CONSUME
        global COLLECT_LOCKER
        COLLECT_LOCKER.acquire()
        if READY_TO_CONSUME == False:
            new_data = self.rssCollect.treatNewsFeedList()
            new_data = [article for _ , article in new_data]
            WINDOW_DATA += new_data
            self.count += 1
        if self.count == self.nb_loop:
            READY_TO_CONSUME = True
            self.count = 0
        COLLECT_LOCKER.release()

    def run(self):
        schedule.every(self.loop_delay).minutes.do(self.update_window_data)
        while True:
            schedule.run_pending()


class NoveltiesDetectionThread(Thread):
    """
    Service to detect novelties in the collect information flow
    """
    def __init__(self, supervised_calculator : SupervisedSequantialLangageSimilarityCalculator , training_args : Dict ,
                 comparaison_args : Dict, loop_delay : int , classifier_models :List[WindowClassifierModel] ):
        Thread.__init__(self)
        self.comparaison_args = comparaison_args
        self.training_args = training_args
        self.supervised_reference_calculator = supervised_calculator
        self.classifier_models = classifier_models
        self.loop_delay = loop_delay


    @staticmethod
    def log_error():
        print("no articles collected during his windows")

    def process(self ):
        global WINDOW_DATA
        global READY_TO_CONSUME
        global PROCESS_LOCKER
        PROCESS_LOCKER.acquire()
        if READY_TO_CONSUME == True:
            if len(WINDOW_DATA) != 0:
                data = transformS(WINDOW_DATA, process_already_done=True)
                self.supervised_reference_calculator.treat_Window(data, **self.training_args)
                self.supervised_reference_calculator.print_novelties(n_to_print=10)
                similarities , _ = self.supervised_reference_calculator.calcule_similarity_topics_W_W(**self.comparaison_args)
                print("@" * 30)
                for topic_id , (similarity_score , classifier) in enumerate(zip(similarities , self.classifier_models)):
                    group_id = classifier.predict(similarity_score)
                    classifier.update(similarity_score)
                    topic = self.supervised_reference_calculator.labels_idx[topic_id]
                    print(f"Rarety level : {group_id} / {len(classifier)} for the topic: {topic}.  ")
                # empty and reinitialization of WINDOW_DATA
                del WINDOW_DATA
                WINDOW_DATA = []
                READY_TO_CONSUME = False
            else:
                NoveltiesDetectionThread.log_error()
        PROCESS_LOCKER.release()


    def run(self):
        schedule.every(self.loop_delay).minutes.do(self.process)
        while True:
            schedule.run_pending()


def startServer():
    '''
    Starts server
    :return:
    '''
    global RSS_FEEDS_PATH
    global OUTPUT_PATH
    global LOOP_DELAY_COLLECT
    stuff : Dict = load_stuff()
    extractor = CollectThread(RSS_FEEDS_PATH,OUTPUT_PATH,LOOP_DELAY_COLLECT)
    detector = NoveltiesDetectionThread(**stuff)
    extractor.start()
    detector.start()
    extractor.join()
    detector.join()
    print("Running Rest RSSNewsExtractor server")


def startAPIs():
    blueprint = Blueprint("api", __name__, url_prefix="/api/v1")
    api = Api(
        blueprint,
        version="1.0",
        validate=False,
    )
    injected_object = {'rss_feed_path': RSS_FEEDS_PATH}
    # inject the objects containing logic here
    for res in namesp.resources:
        res.kwargs['resource_class_kwargs'] = injected_object
        print(res)
    # finally add namespace to api
    api.add_namespace(namesp)
    app = Flask('test')
    app.register_blueprint(blueprint)
    app.run(HOST, port=PORT, debug=True)


if __name__ == '__main__':
    # rootDir = ''
    # if len(sys.argv) > 1:
    #     rootDir=sys.argv[1]
    # with open(os.path.join(rootDir, "../config/config_service.json"), "r") as f:
    #     config = json.load(f)


    startAPIs()
