from typing import List, Dict
from flask import Flask, request
from flask_restplus import Api, Resource, fields
import json
import sys
import os
from threading import Thread , Lock
import schedule
from RSSCollector import RSSCollect
from Sequential_Module import MetaSequencialLangageSimilarityCalculator , SupervisedSequantialLangageSimilarityCalculator
from WindowClassification import WindowClassifierModel

# Creation of the service with Flask
app = Flask(__name__)
api = Api(app)
name_space = api.namespace('RSSNewsExtractor', description='Extract news from RSS feeds and store them into JSON file')

# Model for required data to request the server
url_element = api.model("RSS URL",
                        {"url": fields.String(required=True,
                                              description="URL of the RSS feed to add",
                                              help="URL cannot be empty"),
                         "label": fields.List(fields.String(required=False,
                                                            description="Type(s) of the RSS feed"))})
model = api.model('RSSNewsExtractor Model',
                  {'rss_feed': fields.List(fields.Nested(url_element))})


@api.route("/AddRSSFeedSource")
class RSSNewsExtractor(Resource):
    """
    Rest interface to add rss feed to the extractor
    """
    def get(self):
        return 'DELETE Not available for this service', 404

    @api.expect(model)
    def post(self):
        try:
            # Add rss feed to existing rss feed list
            with open("rssfeed_FUN.json", "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["rss_feed_url"] = rss_feed_url["rss_feed_url"] + request.json['rss_feed']
            with open("rssfeed_FUN.json", "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    @api.expect(model)
    def put(self):
        try:
            # Add rss feed to existing rss feed list
            with open("rssfeed_FUN.json", "r") as f:
                rss_feed_url = json.load(f)
            rss_feed_url["rss_feed_url"] = rss_feed_url["rss_feed_url"] + request.json['rss_feed']
            with open("rssfeed_FUN.json", "w") as f:
                f.write(json.dumps(rss_feed_url))
        except KeyError as e:
            name_space.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            name_space.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    def delete(self):
        return 'DELETE Not available for this service', 404


WINDOW_DATA = []
READY_TO_CONSUME = False
COLLECT_LOCKER = Lock()
PROCESS_LOCKER = Lock()


def initialize_calculator(kwargs_calculator):
    supervised_calculator_type = kwargs_calculator['initialize_engine']['model_type']
    training_args = kwargs_calculator['initialize_engine']['training_args']
    comparaison_args = kwargs_calculator['generate_result']
    del kwargs_calculator['initialize_engine']['model_type']
    del kwargs_calculator['initialize_engine']['training_args']
    sequential_model = supervised_calculator_type
    supervised_calculator: MetaSequencialLangageSimilarityCalculator = sequential_model(
        **kwargs_calculator['initialize_engine'])
    return {
        "supervised_calculator" : supervised_calculator ,
        "comparaison_args" : comparaison_args ,
        "training_args"  : training_args}


class CollectThread(Thread):
    """
    Service to periodically collect information from the selected sources
    """
    def __init__(self,rss_feed_config_file,output_path , delta):
        Thread.__init__(self)
        self.delta = delta * 60 #convert hours to minutes
        self.rss_feed_config=rss_feed_config_file
        self.output_path=output_path
        self.rssCollect=RSSCollect(self.rss_feed_config, self.output_path)
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

    def process(self , data_window):
        global WINDOW_DATA
        global READY_TO_CONSUME
        global PROCESS_LOCKER
        PROCESS_LOCKER.acquire()
        if READY_TO_CONSUME == True:
            if len(WINDOW_DATA) != 0:
                self.supervised_reference_calculator.treat_Window(data_window, **self.training_args)
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
    stuff = initialize_calculator(**config["claculator_initialization"])
    config["service"].update(stuff)
    extractor = CollectThread(config["rss_feed_config_file"],config["output_path"],config["loop_delay"])
    detector = NoveltiesDetectionThread(**config["service"])
    extractor.start()
    detector.start()
    extractor.join()
    detector.join()
    print("Running Rest RSSNewsExtractor server")
    app.run(config["host"], port=config["port"], debug=False)

if __name__ == '__main__':
    rootDir = ''
    if len(sys.argv) > 1:
        rootDir=sys.argv[1]
    with open(os.path.join(rootDir, "config/config_service.json"), "r") as f:
        config = json.load(f)

    startServer()
