from threading import Thread , Lock
from typing import List , Dict
import schedule
from novelties_detection.Collection.RSSCollector import RSSCollect
from novelties_detection.Experience.Sequential_Module import SupervisedSequantialLangageSimilarityCalculator , NoSupervisedSequantialLangageSimilarityCalculator
from novelties_detection.Experience.WindowClassification import WindowClassifierModel
from novelties_detection.Collection.data_processing import transformS
from novelties_detection.Experience.Exception_utils import CompareWindowsException
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)


WINDOW_DATA = []
COLLECT_IN_PROGRESS = False
PROCESS_IN_PROGRESS = False
COLLECT_LOCKER = Lock()
PROCESS_LOCKER = Lock()


def log_function(func):
    def wrapper():
        start_time = datetime.now()
        logging.info(f"{func.__name__} begin at {start_time}")
        func()
        end_time = datetime.now()
        logging.info(f"{func.__name__} finish at {end_time}  , collect duration : {start_time - end_time}")
    return wrapper


class CollectThread(Thread):
    """
    Service to periodically collect information from the selected sources
    """
    def __init__(self,rss_feed_config_file,output_path ,processor, loop_delay):
        Thread.__init__(self)
        self.loop_delay = loop_delay
        self.rss_feed_config=rss_feed_config_file
        self.output_path=output_path
        self.rssCollect=RSSCollect(self.rss_feed_config, self.output_path , processor=processor)

    @log_function
    def update_window_data(self):
        global WINDOW_DATA
        global COLLECT_LOCKER
        global PROCESS_LOCKER
        global COLLECT_IN_PROGRESS
        global PROCESS_IN_PROGRESS
        COLLECT_LOCKER.acquire()
        if PROCESS_IN_PROGRESS == False:
            COLLECT_IN_PROGRESS = True
            new_data = self.rssCollect.treatNewsFeedList()
            WINDOW_DATA += new_data
            COLLECT_IN_PROGRESS = False
        COLLECT_LOCKER.release()

    def run(self):
        logging.info("Collect will start")
        schedule.every(self.loop_delay).minutes.do(self.update_window_data)
        while True:
            schedule.run_pending()


class NoveltiesDetectionThread(Thread):
    """
    Service to detect novelties in the collect information flow
    """
    def __init__(self, supervised_calculator : SupervisedSequantialLangageSimilarityCalculator
                 , micro_calculator : NoSupervisedSequantialLangageSimilarityCalculator , training_args : Dict ,
                 comparaison_args : Dict, micro_training_args : Dict,  loop_delay : int , classifier_models :List[WindowClassifierModel] ):
        Thread.__init__(self)
        self.micro_training_args = micro_training_args
        self.comparaison_args = comparaison_args
        self.training_args = training_args
        self.supervised_reference_calculator = supervised_calculator
        self.micro_calculator = micro_calculator
        self.classifier_models = classifier_models
        self.loop_delay = loop_delay


    @staticmethod
    def log_error():
        logging.warning("no articles collected during his windows")


    def process(self , window_data):
        supervised_data = transformS(window_data, process_already_done=True)
        no_supervised_data = supervised_data[0] #just take the text
        # we train the micro calculator but we don't print the result
        _,_  = self.micro_calculator.treat_Window(no_supervised_data , **self.micro_training_args)
        _,_ = self.supervised_reference_calculator.treat_Window(supervised_data, **self.training_args)
        self.print_res()



    def print_res(self):

        try:
            self.supervised_reference_calculator.print_novelties(n_words_to_print=10 , **self.comparaison_args)
            # we call the function calcul_similarity_topics_W_W in the function print_novelties above but we keep the
            # result in cache so it's more comprehensive to recall the function another time bellow
            new_window_idx = len(self.supervised_reference_calculator)  -1
            previous_window_idx = new_window_idx - 1
            similarities, _ = self.supervised_reference_calculator.calcule_similarity_topics_W_W(previous_window_idx , new_window_idx , **self.comparaison_args)
            print("@" * 30)
            for topic_id, (similarity_score, classifier) in enumerate(zip(similarities, self.classifier_models)):
                classifier.print(similarity_score)
                group_id = classifier.predict(similarity_score)
                classifier.update(similarity_score)
                topic = self.supervised_reference_calculator.labels_idx[topic_id]
                print(f"Rarety level : {group_id} / {len(classifier)} for the topic: {topic}.  ")
        except CompareWindowsException:
            logging.warning("no comparaison possible yet because there are less than 2 windows treated")
            pass
        except Exception as e:
            pass

    @log_function
    def do_process(self):
        global WINDOW_DATA
        global PROCESS_LOCKER
        global PROCESS_IN_PROGRESS
        global COLLECT_IN_PROGRESS
        PROCESS_LOCKER.acquire()
        if COLLECT_IN_PROGRESS == False:
            PROCESS_IN_PROGRESS = True
            if len(WINDOW_DATA) != 0:
                self.process(WINDOW_DATA)
                # empty and reinitialization of WINDOW_DATA
                del WINDOW_DATA
                WINDOW_DATA = []
            else:
                NoveltiesDetectionThread.log_error()
            PROCESS_IN_PROGRESS = False
        PROCESS_LOCKER.release()


    def run(self):
        logging.info("Process will start")
        schedule.every(self.loop_delay).minutes.do(self.do_process)
        while True:
            schedule.run_pending()